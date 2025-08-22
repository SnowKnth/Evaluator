import logging
from typing import DefaultDict, Dict, List, Optional, Tuple, NamedTuple

from evaluator.agent import MobileAgent

from droidbot.input_event import OracleEvent
from .evaluator import BaseEvaluator, FailedReason
from .task_trace import EssentialStateKeyword, TaskTrace, UIState
from .testbed_evaluation.exact_match import (
    check_activity_match,
    check_click_match,
    check_type_match,
    check_uicomponent_match,
    check_page_image_match,
    check_oracle_in_uicomponents,
)
from .testbed_evaluation.fuzzy_match import check_fuzzy_match, compare_entire_ui_vh
from .testbed_evaluation.system_state_match import (
    check_install_match,
    check_uninstall_match,
)

class OracleHitTuple(NamedTuple):
    """
    A tuple representing a hit in the oracle trace.
    - action: The action taken in the oracle trace.
    - ui_state: The UI state at the time of the action.
    - timestamp: The timestamp of the action.
    """
    episode: str
    oracle_dict:  DefaultDict[EssentialStateKeyword, List[str]]
    activity_hit: int = 0
    fuzzy_pageimg_hit: int = 0
    fuzzy_page_main_uicomps_hit: int = 0
    uicomponent_set: int = 0 # = 1 if groundtruth assertion uicomponent is set: {exact<i>, fuzzy<i>, i>=0} 
    uicomponent_nearFull_hit: int = 0 # uicomp_nearFull_hit_list is not empty
    uicomponent_keywords_hit: int = 0
    uicomponent_text_hit: int = 0
    uicomp_nearFull_hit_list: List[int] = None
    uicomp_keywords_hit_list: List[int] = None
    uicomp_text_hit_list: List[int] = None


    


class TestbedEvaluator(BaseEvaluator):
    def __init__(
        self,
        agent: MobileAgent,
        epi_metadata_path: str,
        gr_dataset_path: str,
        options: Dict = None,
    ) -> None:
        super().__init__(agent, epi_metadata_path, gr_dataset_path, options)
        self.evaluator_name = self.__class__.__name__
        self.hit_results: List[OracleHitTuple] = []

        """Ablation Study
        1. fuzzy_match
            - screen_level_fuzzy_match
            - textbox_fuzzy_match
        2. exact_match
            - activity_exact_match
            - action_exact_match
            - UI_component_exact_match
            - system_state_exact_match
        close [fuzzy_match] means close all related to fuzzy match
        close [exact_match] means close all related to exact match
        open [screen_level_fuzzy_match, etc.] means only open itself
        """
        self.fuzzy_match = options.get("fuzzy_match", True) if options else True
        self.exact_match = options.get("exact_match", True) if options else True
        self.screen_level_fuzzy_match = (
            options.get("screen_level_fuzzy_match", False) if options else False
        )
        self.textbox_fuzzy_match = (
            options.get("textbox_fuzzy_match", False) if options else False
        )
        self.activity_exact_match = (
            options.get("activity_exact_match", False) if options else False
        )
        self.action_exact_match = (
            options.get("action_exact_match", False) if options else False
        )
        self.UI_component_exact_match = (
            options.get("UI_component_exact_match", False) if options else False
        )
        self.system_state_exact_match = (
            options.get("system_state_exact_match", False) if options else False
        )

        if (
            self.screen_level_fuzzy_match
            or self.textbox_fuzzy_match
            or self.activity_exact_match
            or self.action_exact_match
            or self.UI_component_exact_match
            or self.system_state_exact_match
        ):
            assert (
                self.screen_level_fuzzy_match
                + self.textbox_fuzzy_match
                + self.activity_exact_match
                + self.action_exact_match
                + self.UI_component_exact_match
                + self.system_state_exact_match
                <= 1
            ), "Only one ablation study can be enabled"
        if self.fuzzy_match:
            self.screen_level_fuzzy_match = True
            self.textbox_fuzzy_match = True
        if self.exact_match:
            self.activity_exact_match = True
            self.action_exact_match = True
            self.UI_component_exact_match = True
            self.system_state_exact_match = True

        self.logger = logging.getLogger(self.evaluator_name)

    def eval_impl(
        self, episode, task_description
    ) -> Tuple[bool, Optional[FailedReason]]:
        gr_trace: TaskTrace = self.helper.load_groundtruth_trace_by_episode(episode)
        if not gr_trace:
            return False, FailedReason.GR_TRACE_NOT_FOUND
        exec_trace: TaskTrace = self.agent.load_exec_trace_by_episode(episode)
        if not exec_trace:
            return False, FailedReason.EXEC_TRACE_NOT_FOUND

        # index for iterating exec_trace
        i = 0

        for ui_state in gr_trace:
            # if the current UIState contains no essential state, go to the next
            if ui_state.essential_state is None:
                continue

            # in this case, there is at least on essential state that is not matched
            # but the exec_traec has been iterated to the end of the list,
            # indicating the task is not completed
            if i == len(exec_trace):
                return False, "Remaining essential states are not matched"

            gr_ui_state_matched = False
            # when there is remaining UIState in the ground-truth trace and
            # remaining UIState in the task exec trace, compare and find two
            # matched UIState
            while i < len(exec_trace):
                cur_exec_ui_state: UIState = exec_trace[i]

                if not self.check_essential_state_match(ui_state, cur_exec_ui_state):
                    # current UIState in the exec trace does not match the
                    # essential state, go to the next UIState in the exec trace
                    i += 1
                    continue
                else:
                    # current essential state matches UIState in the exec trace,
                    # go to the next essential state in the ground-truth trace
                    # and the next UIState in the exec trace
                    gr_ui_state_matched = True
                    # if calc_hit_results is not called, i += 1 is needed
                    i = self.calc_hit_results(ui_state, exec_trace, i)
                    break

            if gr_ui_state_matched:
                continue
            else:
                return False, None

        return True, None

    def dump_hit_results(self, filename: str = "oracle_hit_results.csv"):
        import csv
        if not self.hit_results:
            return
        # 获取所有字段名
        fieldnames = self.hit_results[0]._fields
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tup in self.hit_results:
                # NamedTuple转dict
                row = tup._asdict()
                # 列表字段转字符串
                for k, v in row.items():
                    if isinstance(v, list):
                        row[k] = ",".join(str(x) for x in v)
                writer.writerow(row)
    
    def calc_hit_results(
        self, gr_ui_state: UIState, exec_trace: TaskTrace, node_id: int
    ) -> int:
        """
        Calculate the hit results for the current ground-truth UIState.
        Update the hit_results list with the OracleHitTuple.
        """
        episode: str = gr_ui_state.episode
        activity_hit: int = 0
        fuzzy_pageimg_hit: int = 0
        fuzzy_page_main_uicomps_hit: int = 0
        uicomponent_nearFull_hit: int = 0 # equal to exact hit
        uicomponent_keywords_hit: int = 0
        uicomponent_text_hit: int = 0
        uicomponent_set: int = 0 # = 1 if groundtruth has assertion uicomponents: {exact<i>, fuzzy<i>, i>=0} 
        uicomp_nearFull_hit_list: List[int] = []
        uicomp_keywords_hit_list: List[int] = []
        uicomp_text_hit_list: List[int] = []

        es_dict = gr_ui_state.essential_state
        first_cmp = True
        exec_ui_state = exec_trace[node_id]       
        
        if not exec_ui_state.action.isinstance(OracleEvent):
            return node_id + 1
        
        while node_id < len(exec_trace) and exec_ui_state.action.isinstance(OracleEvent):
            if first_cmp:
                if check_activity_match(gr_ui_state, exec_ui_state):
                    activity_hit = 1
                if check_page_image_match(gr_ui_state.screenshot_path, exec_ui_state.screenshot_path):
                    fuzzy_pageimg_hit = 1
                if compare_entire_ui_vh(gr_ui_state, exec_ui_state):
                    fuzzy_page_main_uicomps_hit = 1
                first_cmp = False
            uicomponent_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.EXACT, None
            )
            fuzzy_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.FUZZY, None
            )
            match_filter: List[int] = []
            if uicomponent_match_states:
                for node_id_str in uicomponent_match_states:
                    node_id_int = int(node_id_str)
                    match_filter.append(node_id_int)

            if fuzzy_match_states:
                for node_id_str in fuzzy_match_states:
                    node_id_int = int(node_id_str)
                    if node_id_int >= 0:
                        match_filter.append(node_id_int)
            if match_filter:
                uicomponent_set = 1
                # judge whether exec_ui_state.action is in dict related to node in match_filter
                nearFull_match, keyword_match, text_match = check_oracle_in_uicomponents(gr_ui_state, exec_ui_state.action, match_filter)
                if nearFull_match != -1:
                    uicomponent_nearFull_hit = 1
                    uicomp_nearFull_hit_list.append(nearFull_match)
                if keyword_match != -1:
                    uicomponent_keywords_hit = 1
                    uicomp_keywords_hit_list.append(keyword_match)
                if text_match != -1:
                    uicomponent_text_hit = 1
                    uicomp_text_hit_list.append(text_match)    
            node_id += 1
            exec_ui_state = exec_trace[node_id]
                
        oracle_hit_tuple = OracleHitTuple(
            episode=episode,
            oracle_dict=gr_ui_state.essential_state,
            activity_hit=activity_hit,
            fuzzy_pageimg_hit=fuzzy_pageimg_hit,
            fuzzy_page_main_uicomps_hit=fuzzy_page_main_uicomps_hit,
            uicomponent_nearFull_hit=uicomponent_nearFull_hit,
            uicomponent_keywords_hit=uicomponent_keywords_hit,
            uicomponent_text_hit=uicomponent_text_hit,
            uicomponent_set=uicomponent_set,
            uicomp_nearFull_hit_list=uicomp_nearFull_hit_list,
            uicomp_keywords_hit_list=uicomp_keywords_hit_list,
            uicomp_text_hit_list=uicomp_text_hit_list,
        )
        self.hit_results.append(oracle_hit_tuple) # key operation
        return node_id

        


    def check_essential_state_match(  # In one gr_ui_state: UIState (.ess file), there are multiple essential keywords in some conditions
        self, gr_ui_state: UIState, exec_ui_state: UIState
    ) -> bool:
        assert (
            gr_ui_state.essential_state is not None
        ), f"essential state in {gr_ui_state} is None"

        es_dict = gr_ui_state.essential_state

        if True:  # fuzzy_match
            fuzzy_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.FUZZY, None
            )
            if fuzzy_match_states and not check_fuzzy_match(
                gr_ui_state,
                exec_ui_state,
                self.screen_level_fuzzy_match,
                self.textbox_fuzzy_match,
            ):
                return False

        if True:  # exact_match:
            uicomponent_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.EXACT, None
            )
            activity_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.ACTIVITY, None
            )
            click_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.CLICK, None
            )
            type_match_states: List[str] = es_dict.get(EssentialStateKeyword.TYPE, None)

            if (
                self.activity_exact_match
                and activity_match_states
                and not check_activity_match(gr_ui_state, exec_ui_state)
            ):
                return False

            if (
                self.UI_component_exact_match
                and uicomponent_match_states
                and not check_uicomponent_match(gr_ui_state, exec_ui_state)
            ):
                return False

            if (
                self.action_exact_match
                and type_match_states
                and not check_type_match(gr_ui_state, exec_ui_state)
            ):
                return False

            if (
                self.action_exact_match
                and click_match_states
                and not check_click_match(gr_ui_state, exec_ui_state)
            ):
                return False

        if self.system_state_exact_match:
            install_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.CHECK_INSTALL, None
            )
            uninstall_match_states: List[str] = es_dict.get(
                EssentialStateKeyword.CHECK_UNINSTALL, None
            )

            if install_match_states and not check_install_match(
                gr_ui_state, exec_ui_state
            ):
                return False

            if uninstall_match_states and not check_uninstall_match(
                gr_ui_state, exec_ui_state
            ):
                return False

        return True
