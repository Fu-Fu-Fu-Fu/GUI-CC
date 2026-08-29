from __future__ import annotations

import unittest
from types import SimpleNamespace

from PIL import Image

from online.actions import plan_to_semantic_action
from online.agent import PlannerAgent, _plan_error
from online.judges import action_description


class PlanValidationTest(unittest.TestCase):
    def test_judge直接使用完整低层动作描述(self) -> None:
        action = {
            "type": "tap",
            "target": "Click on the Save button.",
            "low_level_instruction": "Click on the Save button.",
        }
        self.assertEqual(action_description(action), "Click on the Save button.")

    def test_plan要求动作专用字段(self) -> None:
        self.assertIsNotNone(_plan_error({"type": "scroll", "x": 10, "y": 20}))
        self.assertIsNotNone(_plan_error({"type": "tap", "target": "Save"}))
        self.assertIsNotNone(_plan_error({"type": "open_app"}))
        self.assertIsNotNone(_plan_error({"type": "terminate", "status": "done"}))
        self.assertIsNone(_plan_error({
            "type": "scroll", "x": 10, "y": 20, "direction": "up"
        }))
        self.assertIsNone(_plan_error({"type": "tap", "x": 10, "y": 20}))


class PlanToActionTest(unittest.TestCase):
    def test_像素坐标被裁剪到画面内(self) -> None:
        action = plan_to_semantic_action({"type": "tap", "x": 9999, "y": -5}, (100, 200))
        self.assertEqual(action["source_coord"], [99, 0])

    def test_scroll保留手指方向并派生起止点(self) -> None:
        action = plan_to_semantic_action(
            {"type": "scroll", "x": 50, "y": 100, "direction": "up"}, (100, 200))
        self.assertEqual(action["direction"], "up")
        # 手指向上：起点在下、终点在上
        self.assertGreater(action["start_norm"][1], action["end_norm"][1])

    def test_终止动作保留状态与文本(self) -> None:
        terminate = plan_to_semantic_action(
            {"type": "terminate", "status": "success"}, (100, 200))
        self.assertEqual(terminate["value"], "success")
        answer = plan_to_semantic_action(
            {"type": "answer", "answer_text": "done"}, (100, 200))
        self.assertEqual(answer["text"], "done")


class AgentStepTest(unittest.TestCase):
    @staticmethod
    def _agent(plan: dict) -> PlannerAgent:
        agent = PlannerAgent.__new__(PlannerAgent)
        agent.action_history = []
        agent._call_planner = lambda _frame: plan
        return agent

    def test_不支持的planner动作会报错(self) -> None:
        agent = self._agent({"type": "invent_action"})
        raw, action = agent.step(Image.new("RGB", (32, 64)))
        self.assertIn("unsupported_action", raw)
        self.assertIsNone(action)
        self.assertEqual(agent.last_step_evidence["failure"]["kind"], "planner_unsupported_action")

    def test_planner无回复时不产出动作(self) -> None:
        agent = self._agent(None)
        raw, action = agent.step(Image.new("RGB", (32, 64)))
        self.assertIsNone(action)
        self.assertIn("planner_response_parse_failed", raw)

    def test_正常plan产出语义动作(self) -> None:
        agent = self._agent({"type": "tap", "x": 10, "y": 20, "target": "Save"})
        raw, action = agent.step(Image.new("RGB", (32, 64)))
        self.assertEqual(action["type"], "tap")
        self.assertEqual(action["source_coord"], [10, 20])
        self.assertIn("tap on Save", raw)


def _fake_response(model: str, *, tool_name: str | None = None,
                   arguments: str = "", content: str | None = None):
    """构造一个 OpenAI chat.completions 形状的假响应。"""
    tool_calls = None
    if tool_name is not None:
        tool_calls = [SimpleNamespace(
            function=SimpleNamespace(name=tool_name, arguments=arguments))]
    message = SimpleNamespace(tool_calls=tool_calls, content=content)
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=message)], usage=None)


class CallPlannerTest(unittest.TestCase):
    """_call_planner 的原生 tool-call 解析路径，client 是假的，不发请求。"""

    @staticmethod
    def _agent(response) -> PlannerAgent:
        agent = PlannerAgent.__new__(PlannerAgent)
        agent.planner_model = "planner-x"
        agent.planner_temperature = 1.0
        agent.planner_max_tokens = 128
        agent.max_retry = 1  # 只试一次，失败分支不会 sleep
        agent.instruction = "打开设置"
        agent.action_history = []
        agent.last_step_evidence = {"planner_attempts": []}
        agent.planner = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)))
        return agent

    def _call(self, agent: PlannerAgent):
        return agent._call_planner(Image.new("RGB", (32, 64)))

    def test_computer工具调用被解析成plan(self) -> None:
        agent = self._agent(_fake_response(
            "planner-x", tool_name="computer",
            arguments='{"type": "tap", "x": 10, "y": 20, "target": "Save"}'))
        plan = self._call(agent)
        self.assertEqual(plan["type"], "tap")
        self.assertEqual((plan["x"], plan["y"]), (10, 20))
        attempt = agent.last_step_evidence["planner_attempts"][0]
        self.assertIsNone(attempt["validation_error"])
        self.assertEqual(attempt["response_model"], "planner-x")
        self.assertNotIn("failure", agent.last_step_evidence)

    def test_coordinate数组是主路径(self) -> None:
        agent = self._agent(_fake_response(
            "planner-x", tool_name="computer",
            arguments='{"type": "tap", "coordinate": [10, 20], "target": "Save"}'))
        plan = self._call(agent)
        self.assertEqual((plan["x"], plan["y"]), (10, 20))

    def test_坐标数组塞进x字段也能收(self) -> None:
        # 实测 qwen3.7-plus 面对 x/y schema 时会把 [x, y] 塞进 x；容错并归一
        agent = self._agent(_fake_response(
            "planner-x", tool_name="computer",
            arguments='{"type": "tap", "x": [940, 56], "target": "Search"}'))
        plan = self._call(agent)
        self.assertEqual((plan["x"], plan["y"]), (940, 56))

    def test_模型没调工具时无plan(self) -> None:
        agent = self._agent(_fake_response("planner-x", content="我先想想再动手。"))
        self.assertIsNone(self._call(agent))
        self.assertEqual(agent.last_step_evidence["failure"]["kind"], "planner_no_tool_call")

    def test_工具参数不是合法JSON时无plan(self) -> None:
        agent = self._agent(_fake_response(
            "planner-x", tool_name="computer", arguments='{"type": "tap", '))
        self.assertIsNone(self._call(agent))
        failure = agent.last_step_evidence["failure"]
        self.assertEqual(failure["kind"], "planner_validation_failed")
        self.assertIn("不是合法 JSON", failure["message"])

    def test_动作类型不在允许集合时无plan(self) -> None:
        agent = self._agent(_fake_response(
            "planner-x", tool_name="computer", arguments='{"type": "invent_action"}'))
        self.assertIsNone(self._call(agent))
        self.assertEqual(agent.last_step_evidence["failure"]["kind"],
                         "planner_unsupported_action")

    def test_响应模型与请求模型不一致时无plan(self) -> None:
        agent = self._agent(_fake_response(
            "另一个模型", tool_name="computer",
            arguments='{"type": "tap", "x": 1, "y": 2}'))
        self.assertIsNone(self._call(agent))
        failure = agent.last_step_evidence["failure"]
        self.assertEqual(failure["kind"], "planner_model_identity_error")
        self.assertIn("另一个模型", failure["message"])


if __name__ == "__main__":
    unittest.main()
