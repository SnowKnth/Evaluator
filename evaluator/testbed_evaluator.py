import logging
from typing import Dict, List

from evaluator.agent import MobileAgent

from .common.action_type import Action
from .evaluator import BaseEvaluator
from .task_trace import (
    TaskTrace,
    get_all_actions,
    get_all_screenshot_paths,
    get_all_vh_paths,
)
from .testbed_evaluation.comparison_algorithm import comparison_algorithm
from .testbed_evaluation.get_crucial_states import CrucialState, CrucialStates
from .testbed_evaluation.xml_exactly_match import exactly_match
from .testbed_evaluation.xml_fuzzy_match import get_xml_fuzzy_match


class TestbedEvaluator(BaseEvaluator):
    def __init__(self, agent: MobileAgent, options: Dict = None) -> None:
        super().__init__(agent, options)
        self.evaluator_name = self.__class__.__name__
        self.logger = logging.getLogger(self.evaluator_name)

    def load_crucial_states_by_episode(self, episode):
        gt_trace = self.helper.load_groundtruth_trace_by_episode(episode)

    def eval_impl(self, episode, task_description) -> bool:
        # load crucial states
        # crucial_states: CrucialStates = self.load_crucial_states_by_episode(episode)

        # # load task execution trace of a specific mobile agent
        # task_exec_trace: TaskTrace = self.agent.load_exec_trace_by_episode(episode)
        # screenshot_paths: List[str] = get_all_screenshot_paths(task_exec_trace)
        # vh_paths: List[str] = get_all_vh_paths(task_exec_trace)
        # actions: List[Action] = get_all_actions(task_exec_trace)

        # # for each crucial states, find matching things


        # load the ground-truth trace and crucial states
        testbed_groudtruth_trace_path = (
            self.helper.load_testbed_goundtruth_trace_path_by_episode(episode)
        )
        gr_trace: TaskTrace = self.helper.load_groundtruth_trace_by_episode(episode)
        gr_vh_paths: List[str] = get_all_vh_paths(gr_trace)
        print(gr_vh_paths)

        cs: CrucialStates = CrucialStates(episode, testbed_groudtruth_trace_path)
        cs_fuzzy_list: List[Dict] = cs.get_fuzzy_match_list()

        # load the task execution trace
        exec_trace: TaskTrace = self.agent.load_exec_trace_by_episode(episode)
        exec_vh_paths: List[str] = get_all_vh_paths(exec_trace)

        for cs_fuzzy_element in cs_fuzzy_list:
            pic_id = int(cs_fuzzy_element["pic_id"])
            node_id = int(cs_fuzzy_element["node_id"])  # TODO: how to node_id is utilized

            gr_vh_path = gr_vh_paths[pic_id]
            print(
                f"<{gr_vh_path}> node<{node_id}> should be fuzzily matched"
            )

            # iterate the vhs in the exec trace
            for exec_vh_path in exec_vh_paths:
                if get_xml_fuzzy_match(
                    checkpoint_xml_path=gr_vh_path,
                    node_id=node_id,
                    captured_xml_path=exec_vh_path,
                    COSINE_BOUND=0.75
                ):
                    print(
                        f"<{gr_vh_path}> fuzzily matches <{exec_vh_path}> with node<{node_id}>"
                    )

                    cs_exact_list = cs.get_pic_exactly_match_list(pic_id)
                    matched_elements = 0
                    for cs_exact_element in cs_exact_list:
                        keyword = cs_exact_element.keyword
                        node_id = cs_exact_element.node_id
                        if exactly_match(
                            checkpoint=cs_exact_element,
                            checkpoint_dir=checkpoint_dir,
                            pic_id=pic_id,
                            keyword=keyword,
                            node_id=node_id,
                            captured_dir=captured_dir,
                            index,
                        ):
                            matched_elements += 1

                    if matched_elements == len(cs_exact_list):
                        # mark all exact match items are matched
                        for cs_exact_element in cs_exact_list:
                            cs_exact_element.matched = True
                            cs_exact_element.capture_id = (
                                exec_vh_path.split("/")[-1].split(".")[0]
                            )

        cs = [state for state in cs.get_crucial_states() if state.keyword != "fuzzy_match"]
        matched_state = [s.matched for s in cs]
        if False in matched_state:
            return False, "Not all crucial states are matched"
        else:
            matched_exec_vh_ids = [int(s.capture_id) for s in cs]
            if matched_exec_vh_ids == sorted(matched_exec_vh_ids):
                return True
            else:
                return False, "All crucial states are matched but with an incorrect order"