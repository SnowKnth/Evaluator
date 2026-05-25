# # Run all 5 agents with all 3 step limits (15 total runs)
# python RASSDroid_Eval.py --eval t --run_all

# # Run a specific agent with all step limits
# python RASSDroid_Eval.py --eval t --agent VASSODroid_No_Validation

# # Run all agents with a specific step limit
# python RASSDroid_Eval.py --eval t --step_limit 20

# # Run a specific agent with a specific step limit
# python RASSDroid_Eval.py --eval t --agent VASSODroid_No_Validation --step_limit 20

import argparse
import os
from typing import List, Optional, Dict

from config import CONFIG
from evaluator.agent import MobileAgent
from evaluator.common.action_type import Action
from evaluator.exactmatch_evaluator import ExactMatchEvaluator
from evaluator.lcsmatch_evaluator import LCSMatchEvaluator
from evaluator.task_trace import Agent, DatasetHelper, TaskCategory, TaskTrace
from evaluator.testbed_evaluator import TestbedEvaluator


# Agent configurations: agent_name -> (Agent enum, exec_trace_path)
AGENT_CONFIGS: Dict[str, tuple] = {
    # DeepSeek V3.2 (R1) versions
    "VASSODroid_Full_Follow_and_Adapt_InTime": (Agent.VASSODroid_Full_Follow_and_Adapt_InTime, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FOLLOW_AND_ADAPT_INTIME),
    "VASSODroid_No_VASSO": (Agent.VASSODroid_No_VASSO, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_NO_VASSO),
    "VASSODroid_Full_First_Follow_then_Adapt": (Agent.VASSODroid_Full_First_Follow_then_Adapt, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FIRST_FOLLOW_THEN_ADAPT),
    "VASSODroid_No_Interaction_Validation": (Agent.VASSODroid_No_Interaction_Validation, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_NO_INTERACTION_VALIDATION),
    "AutoDroid": (Agent.AutoDroid, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_AUTODROID),

    # DeepSeek V3 versions
    "VASSODroid_Full_Follow_and_Adapt_InTime_DSV3": (Agent.VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FOLLOW_AND_ADAPT_INTIME_DSV3),
    "VASSODroid_Full_First_Follow_then_Adapt_DSV3": (Agent.VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3, CONFIG.RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FIRST_FOLLOW_THEN_ADAPT_DSV3),
}

# Step limits to evaluate
STEP_LIMITS = [10, 20, 30]


class VASSODroid(MobileAgent):
    def __init__(self, agent_name: str = "VASSODroid_No_Validation", no_oracle_step_limit: int = 30) -> None:
        super().__init__()
        
        if agent_name not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent: {agent_name}. Available agents: {list(AGENT_CONFIGS.keys())}")
        
        agent_enum, exec_trace_path = AGENT_CONFIGS[agent_name]
        
        self.agent_name = agent_name
        self.no_oracle_step_limit = no_oracle_step_limit
        self.agent = agent_enum
        self.agent_exec_trace_path = exec_trace_path
        
    def load_predicted_action_by_episode(self, episode: str) -> Optional[List[Action]]:
        '''extracts the action sequence from an agent execution trace. This is used for the two baseline evaluation approaches involving only action match.'''
        exec_trace: TaskTrace = self.load_exec_trace_by_episode(episode)
        if exec_trace:
            return [ui_state.action for ui_state in exec_trace]
        return None


    def load_exec_trace_by_episode(self, episode: str) -> Optional[TaskTrace]:
        '''takes a string-format episode as the input, and returns a TaskTrace object containing all recorded information during executing the task on AgentEnv. Agents should have their own implementation for this method, such as specifying the path of agent execution traces.'''
        helper = DatasetHelper(CONFIG.EPI_METADATA_PATH, CONFIG.GR_DATASET_PATH)
        category = helper.get_category_by_episode(episode)
        if category == TaskCategory.WEBSHOPPING:
            category_val = "web_shopping"
        elif category == TaskCategory.GOOGLEAPPS:
            category_val = "google_apps"
        else:
            category_val = category.value
        epi_trace_path = os.path.join(
            self.agent_exec_trace_path, category_val, episode, "captured_data"
        )
        if not os.path.exists(epi_trace_path):
            return None
        return helper.load_testbed_trace_by_path(epi_trace_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RASSDroid evaluation.")
    parser.add_argument(
        "--eval", type=str, help='Evaluation type: "testbed (t)" or "exact (e)"'
    )
    parser.add_argument(
        "--agent", type=str, default=None,
        help=f'Agent name. Available: {list(AGENT_CONFIGS.keys())}. If not specified, run all agents.'
    )
    parser.add_argument(
        "--step_limit", type=int, default=None,
        help=f'Step limit. Available: {STEP_LIMITS}. If not specified, run all step limits.'
    )
    parser.add_argument(
        "--run_all", action="store_true",
        help='Run all agents with all step limits (overrides --agent and --step_limit)'
    )
    args = parser.parse_args()

    def run_testbed_evaluation(agent_name: str, step_limit: int, episodes: List[str] = None):
        """Run testbed evaluation for a specific agent and step limit."""
        print(f"\n{'='*80}")
        print(f"Running Testbed Evaluation: Agent={agent_name}, StepLimit={step_limit}")
        print(f"{'='*80}")
        
        agent = VASSODroid(agent_name=agent_name, no_oracle_step_limit=step_limit)
        
        eval_options = {
            "categories": [
                TaskCategory.GENERAL,
                TaskCategory.GOOGLEAPPS,
                TaskCategory.INSTALL,
                TaskCategory.WEBSHOPPING,
                TaskCategory.GENERATED,
            ],
            "check_fuzzy_match": True,
            "check_exact_match": True,
            "check_system_state": True,
        }
        
        if episodes:
            eval_options["episodes"] = episodes
        
        t = TestbedEvaluator(
            agent=agent,
            epi_metadata_path=CONFIG.EPI_METADATA_PATH,
            gr_dataset_path=CONFIG.GR_DATASET_PATH,
            options=eval_options,
        )
        t.no_oracle_step_limit = step_limit  # Set the step limit in evaluator
        t.run_evaluation()
        t.report_stats(
            human_eval_path=CONFIG.AUTOUI_HUMANEVAL_PATH,
            only_human_eval_positive=False,
        )
        return t

    def run_exact_evaluation(agent_name: str, step_limit: int):
        """Run exact match evaluation for a specific agent and step limit."""
        print(f"\n{'='*80}")
        print(f"Running Exact Match Evaluation: Agent={agent_name}, StepLimit={step_limit}")
        print(f"{'='*80}")
        
        agent = VASSODroid(agent_name=agent_name, no_oracle_step_limit=step_limit)
        
        e = ExactMatchEvaluator(
            agent=agent,
            epi_metadata_path=CONFIG.EPI_METADATA_PATH,
            gr_dataset_path=CONFIG.GR_DATASET_PATH,
            options={
                "categories": [
                    TaskCategory.GENERAL,
                    TaskCategory.GOOGLEAPPS,
                    TaskCategory.INSTALL,
                    TaskCategory.WEBSHOPPING,
                    TaskCategory.GENERATED,
                ],
            },
        )
        e.run_evaluation()
        e.report_stats(
            human_eval_path=CONFIG.AUTOUI_HUMANEVAL_PATH,
            only_human_eval_positive=False,
        )
        return e

    def run_lcs_evaluation(agent_name: str, step_limit: int):
        """Run LCS match evaluation for a specific agent and step limit."""
        print(f"\n{'='*80}")
        print(f"Running LCS Match Evaluation: Agent={agent_name}, StepLimit={step_limit}")
        print(f"{'='*80}")
        
        agent = VASSODroid(agent_name=agent_name, no_oracle_step_limit=step_limit)
        
        t = LCSMatchEvaluator(
            agent=agent,
            epi_metadata_path=CONFIG.EPI_METADATA_PATH,
            gr_dataset_path=CONFIG.GR_DATASET_PATH,
            options={
                "categories": [
                    TaskCategory.GENERAL,
                    TaskCategory.GOOGLEAPPS,
                    TaskCategory.INSTALL,
                    TaskCategory.WEBSHOPPING,
                    TaskCategory.GENERATED,
                ],
            },
        )
        t.run_evaluation()
        t.report_stats(
            human_eval_path=CONFIG.AUTOUI_HUMANEVAL_PATH,
            only_human_eval_positive=False,
        )
        return t

    # Determine which agents and step limits to run
    if args.run_all:
        agents_to_run = list(AGENT_CONFIGS.keys())
        step_limits_to_run = STEP_LIMITS
    else:
        agents_to_run = [args.agent] if args.agent else list(AGENT_CONFIGS.keys())
        step_limits_to_run = [args.step_limit] if args.step_limit else STEP_LIMITS

    # Validate agent names
    for agent_name in agents_to_run:
        if agent_name not in AGENT_CONFIGS:
            raise ValueError(f"Unknown agent: {agent_name}. Available agents: {list(AGENT_CONFIGS.keys())}")

    # Validate step limits
    for step_limit in step_limits_to_run:
        if step_limit not in STEP_LIMITS:
            print(f"Warning: Step limit {step_limit} is not in the predefined list {STEP_LIMITS}")

    print(f"\n{'#'*80}")
    print(f"Evaluation Configuration:")
    print(f"  Evaluation Type: {args.eval}")
    print(f"  Agents: {agents_to_run}")
    print(f"  Step Limits: {step_limits_to_run}")
    print(f"  Total Runs: {len(agents_to_run) * len(step_limits_to_run)}")
    print(f"{'#'*80}")

    # Run evaluations
    results = {}
    
    if args.eval == "exact" or args.eval == "e":
        for agent_name in agents_to_run:
            for step_limit in step_limits_to_run:
                key = f"{agent_name}_step{step_limit}"
                results[key] = run_exact_evaluation(agent_name, step_limit)

    elif args.eval == "testbed" or args.eval == "t":
        for agent_name in agents_to_run:
            for step_limit in step_limits_to_run:
                key = f"{agent_name}_step{step_limit}"
                results[key] = run_testbed_evaluation(agent_name, step_limit)

    elif args.eval == "lcs-exact" or args.eval == "lcse":
        for agent_name in agents_to_run:
            for step_limit in step_limits_to_run:
                key = f"{agent_name}_step{step_limit}"
                results[key] = run_lcs_evaluation(agent_name, step_limit)

    else:
        raise Exception(
            f"Invalid evaluation type: {args.eval}, expected: testbed/t, exact/e, or lcs-exact/lcse"
        )

    # Print summary
    print(f"\n{'#'*80}")
    print("Evaluation Summary")
    print(f"{'#'*80}")
    print(f"Completed {len(results)} evaluation runs:")
    for key in results:
        print(f"  - {key}")
