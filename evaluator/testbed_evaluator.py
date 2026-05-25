import logging
import os
from typing import DefaultDict, Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime

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
    node_start_id: int = -1
    node_start_oracle_id: int = -1
    oracle_dict:  DefaultDict[EssentialStateKeyword, List[str]] = None
    page_hit: int = 0
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

class HitMetrics(NamedTuple):
    pages_annotated_hit: int
    assertion_states_exec_total: int
    assertion_states_gen_total: int
    assertion_states_annotated_total: int
    assertion_states_annotated_hit: int
    is_assert_widget_set: bool

    


class TestbedEvaluator(BaseEvaluator):
    def __init__(
        self,
        agent: MobileAgent,
        epi_metadata_path: str,
        gr_dataset_path: str,
        options: Dict = None,
        no_oracle_step_limit: int = 30,
    ) -> None:
        super().__init__(agent, epi_metadata_path, gr_dataset_path, options)
        self.evaluator_name = self.__class__.__name__
        self.hit_results: List[OracleHitTuple] = []

        self.no_oracle_step_limit = no_oracle_step_limit
        
        self.unerror_episodes_total: int = 0
        self.completed_episodes_count: int = 0
        self.TAR: float = 0.0 # Test Task (Episode) Completion Rate
        
        self.unerror_episodes_total_easy: int = 0
        self.completed_episodes_count_easy: int = 0
        self.TAR_easy: float = 0.0 # Test Task (Episode) Completion Rate
        
        self.unerror_episodes_total_mid: int = 0
        self.completed_episodes_count_mid: int = 0
        self.TAR_mid: float = 0.0 # Test Task (Episode) Completion Rate

        self.unerror_episodes_total_hard: int = 0
        self.completed_episodes_count_hard: int = 0
        self.TAR_hard: float = 0.0 # Test Task (Episode) Completion Rate

        self.key_subtasks_total: int = 0 # total key subtasks marked by pages with one or several assertions
        self.completed_key_subtasks: int = 0 # key subtasks that are completed
        self.KSAR: float = 0.0 # Key Subtask Completion Rate

        self.key_subtasks_total_easy: int = 0 # total key subtasks marked by pages with one or several assertions
        self.completed_key_subtasks_easy: int = 0 # key subtasks that are completed
        self.KSAR_easy: float = 0.0 # Key Subtask Completion Rate

        self.key_subtasks_total_mid: int = 0 # total key subtasks marked by pages with one or several assertions
        self.completed_key_subtasks_mid: int = 0 # key subtasks that are completed
        self.KSAR_mid: float = 0.0 # Key Subtask Completion Rate
        
        self.key_subtasks_total_hard: int = 0 # total key subtasks marked by pages with one or several assertions
        self.completed_key_subtasks_hard: int = 0 # key subtasks that are completed
        self.KSAR_hard: float = 0.0 # Key Subtask Completion Rate

        self.pages_exec_total: int = 0 # total pages in execution trace, denominator(分母) for page-level accuracy
        self.pages_gen_total: int = 0 # total pages with generated assertions, denominator(分母) for page-level precision
        self.pages_annotated_total: int = 0 # total pages with annotated assertions, denominator(分母) for page-level recall
        self.pages_annotated_hit: int = 0 # pages with annotated assertions and at least one hit, numerator(分子) for page-level recall and precision
        self.pages_no_annotation_nor_gen_total: int = 0 # true negative: pages without annotated assertions and without generated assertions; along with pages_annotated_hit, numerator(分子) for page-level accuracy
        
        self.pages_exec_total_easy: int = 0
        self.pages_gen_total_easy: int = 0
        self.pages_annotated_total_easy: int = 0
        self.pages_annotated_hit_easy: int = 0
        self.pages_no_annotation_nor_gen_total_easy: int = 0
        
        self.pages_exec_total_mid: int = 0
        self.pages_gen_total_mid: int = 0
        self.pages_annotated_total_mid: int = 0
        self.pages_annotated_hit_mid: int = 0
        self.pages_no_annotation_nor_gen_total_mid: int = 0
        
        self.pages_exec_total_hard: int = 0
        self.pages_gen_total_hard: int = 0
        self.pages_annotated_total_hard: int = 0
        self.pages_annotated_hit_hard: int = 0
        self.pages_no_annotation_nor_gen_total_hard: int = 0
         
        self.assertion_states_exec_total: int = 0 # total assertion states in execution trace, denominator(分母) for assertion-level accuracy
        self.assertion_states_gen_total: int = 0 # total generated assertion states, denominator(分母) for assertion-level precision
        self.assertion_states_annotated_total: int = 0 # total annotated assertion states, denominator(分母) for assertion-level recall
        self.assertion_states_annotated_hit: int = 0 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
        self.assertion_states_no_annotation_nor_gen_total: int = 0 # true negative: assertion states without annotated assertions and without generated assertions; along with assertion_states_annotated_hit, numerator(分子) for assertion-level accuracy
        
        self.recognized_pages_with_widget_assertions: int = 0 # total recognized pages with annotated widget assertions
        self.recognized_pages_with_widget_assertions_and_evidence_widget_hits: int = 0 # total pages with annotated widget assertions and evidence widget hits
        self.widget_hit_rate: float = 0.0

        self.recognized_pages_with_widget_assertions_easy: int = 0
        self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy: int = 0
        self.widget_hit_rate_easy: float = 0.0
        
        self.recognized_pages_with_widget_assertions_mid: int = 0
        self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid: int = 0
        self.widget_hit_rate_mid: float = 0.0
        
        self.recognized_pages_with_widget_assertions_hard: int = 0
        self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard: int = 0
        self.widget_hit_rate_hard: float = 0.0

        self.total_actions_pos_for_finished_key_subtask: int = 0
        self.average_actions_pos_for_finished_key_subtask: float = 0.0
        
        self.total_actions_pos_for_finished_key_subtask_easy: int = 0
        self.average_actions_pos_for_finished_key_subtask_easy: float = 0.0
        
        self.total_actions_pos_for_finished_key_subtask_mid: int = 0
        self.average_actions_pos_for_finished_key_subtask_mid: float = 0.0

        self.total_actions_pos_for_finished_key_subtask_hard: int = 0
        self.average_actions_pos_for_finished_key_subtask_hard: float = 0.0
        
                #self.assertions_in_recognized_pages_with_widget_assertions: int = 0 # total annotated assertions in recognized pages with annotated widget assertions
        #self.evidence_widget_hits_in_recognized_pages_with_widget_assertions: int = 0 # total evidence widget hits in recognized pages with annotated widget assertions

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
        
        # Determine difficulty level based on gr_trace length
        gr_trace_len = len(gr_trace)
        if gr_trace_len < 4:
            difficulty = "easy"
        elif gr_trace_len < 8:
            difficulty = "mid"
        else:
            difficulty = "hard"
        
        # Track unerror episodes by difficulty
        self.unerror_episodes_total += 1
        if difficulty == "easy":
            self.unerror_episodes_total_easy += 1
        elif difficulty == "mid":
            self.unerror_episodes_total_mid += 1
        else:
            self.unerror_episodes_total_hard += 1
        
        for ui_state in gr_trace:          
            # if the current UIState contains no essential state, go to the next
            if ui_state.essential_state is not None:
                self.key_subtasks_total += 1
                if difficulty == "easy":
                    self.key_subtasks_total_easy += 1
                elif difficulty == "mid":
                    self.key_subtasks_total_mid += 1
                else:
                    self.key_subtasks_total_hard += 1
        
        # ---------------------------------------------------------------------
        # Page-level metrics bookkeeping (KESR at "page" granularity)
        #
        # We treat the execution trace (exec_trace) as an interleaving sequence of:
        #   (1) normal interaction actions (tap/input/scroll/...) and
        #   (2) assertion actions represented by OracleEvent (LLM-generated assertions).
        #
        # IMPORTANT: page counting uses a de-duplication rule:
        #   - Consecutive accepted OracleEvent assertions are considered to belong to the SAME page,
        #     and contribute only +1 to page totals (via `last_is_assert` gating).
        #   - This is because multiple assertions can be generated on the same GUI page/state.
        #
        # Definitions (per current ground-truth essential state segment, accumulated into overall):
        #
        # tmp_pages_exec_total:
        #   "How many pages were visited (counted) in the execution trace?"
        #   - Incremented by 1 when we see a normal (non-OracleEvent) action and the previous
        #     accepted assertion block is not ongoing (`not last_is_assert`).
        #   - Also incremented by 1 when we see the FIRST accepted OracleEvent in a consecutive
        #     assertion block (`OracleEvent && assert_accept && not last_is_assert`).
        #   => Denominator for page-level Accuracy.
        #
        # tmp_pages_gen_total:
        #   "How many pages have generated assertions (predicted positive pages)?"
        #   - Incremented by 1 only on the FIRST accepted OracleEvent of a consecutive assertion block:
        #     (`OracleEvent && assert_accept && not last_is_assert`).
        #   => Denominator for page-level Precision.
        #
        # tmp_pages_annotated_total:
        #   "How many essential pages exist in ground-truth (actual positive pages)?"
        #   - Incremented by 1 when a ground-truth essential UIState is matched by some exec UIState
        #     (i.e., once per matched essential state).
        #   => Denominator for page-level Recall.
        #
        # tmp_pages_annotated_hit:
        #   "How many essential pages are correctly recognized (true positive pages)?"
        #   - Set to 1 (per matched essential state) inside `calc_hit_results()` when there is at least
        #     one accepted assertion on the matched essential page (page_hit).
        #   - NOTE: In eval_impl, this local tmp_pages_annotated_hit is NOT directly updated; instead
        #     we add `hit_metrics.pages_annotated_hit` returned by `calc_hit_results()`.
        #   => Numerator for page-level Precision and Recall.
        #
        # tmp_pages_no_annotation_nor_gen_total:
        #   "How many pages are true negatives (no GT essential page AND no generated assertion)?"
        #   - Incremented when the current exec UIState does NOT match the current GT essential state,
        #     and the exec action is NOT OracleEvent, and `not last_is_assert`.
        #   - Intuition: these are pages we visited that are not essential, and we also did not generate
        #     assertions on them.
        #   => Along with pages_annotated_hit (TP), forms the numerator for page-level Accuracy:
        #      Accuracy = (TP_pages + TN_pages) / pages_exec_total
        #
        # Note on `last_is_assert`:
        #   `last_is_assert` tracks whether the previous exec UIState counted for pages was an accepted
        #   OracleEvent. This is used to ensure multiple consecutive assertions are not double-counted
        #   as multiple pages.
        # 
        # TP_pages = pages_annotated_hit
        # FP_pages = pages_gen_total - pages_annotated_hit
        # FN_pages = pages_annotated_total - pages_annotated_hit
        # TN_pages = pages_no_annotation_nor_gen_total
        # ---------------------------------------------------------------------
        tmp_pages_exec_total: int = 0  # denominator for page-level accuracy
        tmp_pages_gen_total: int = 0  # denominator for page-level precision
        tmp_pages_annotated_total: int = 0  # denominator for page-level recall
        tmp_pages_annotated_hit: int = 0  # numerator for page-level precision/recall (mostly from calc_hit_results)
        tmp_pages_no_annotation_nor_gen_total: int = 0  # TN pages; with TP forms numerator for page-level accuracy
         
        tmp_assertion_states_exec_total: int = 0 # total assertion states in execution trace, denominator(分母) for assertion-level accuracy
        tmp_assertion_states_gen_total: int = 0 # total generated assertion states, denominator(分母) for assertion-level precision
        tmp_assertion_states_annotated_total: int = 0 # total annotated assertion states, denominator(分母) for assertion-level recall
        tmp_assertion_states_annotated_hit: int = 0 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
        tmp_assertion_states_no_annotation_nor_gen_total: int = 0 # true negative: assertion states without annotated assertions and without generated assertions;
        
        last_is_assert = False # whether the last UIState in exec_trace is an accepted assertion state

        # index for iterating exec_trace
        i = 0
        i_no_oracle = 0

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
            while i < len(exec_trace) and i_no_oracle < self.no_oracle_step_limit:
                cur_exec_ui_state: UIState = exec_trace[i]
                
                if cur_exec_ui_state.action is not None: # 是否有效State(含action或assertion)
                    if isinstance(cur_exec_ui_state.action, OracleEvent): # 开头或开始匹配之前遇到的oracle，如果页面-事件对和标注的oracle匹配了会向后遍历，第一个不会是oracle
                        if cur_exec_ui_state.action.assert_accept:
                            tmp_assertion_states_exec_total += 1
                            tmp_assertion_states_gen_total += 1
                            if not last_is_assert: # first accepted assertion in a list (only happen for unannotated states)
                                tmp_pages_exec_total += 1
                                tmp_pages_gen_total += 1
                                last_is_assert_tmp = True
                            else:
                                last_is_assert_tmp = True
                                i += 1
                                continue
                        else:
                            # rejected assertion, skip
                            i += 1
                            continue
                    else: # 普通事件
                        if not last_is_assert:
                            tmp_pages_exec_total += 1                     
                            tmp_assertion_states_exec_total += 1
                        else:
                            pass # 不能continue，有可能是click事件匹配
                        i_no_oracle += 1
                        last_is_assert_tmp = False
                else:
                    i += 1
                    continue
                        
                if not self.check_essential_state_match(ui_state, cur_exec_ui_state):
                    # current UIState in the exec trace does not match the
                    # essential state, go to the next UIState in the exec trace
                    if not isinstance(cur_exec_ui_state.action, OracleEvent):
                        if not last_is_assert:
                            tmp_pages_no_annotation_nor_gen_total += 1 # 必须新页面，没有annotation，也没有generate
                            tmp_assertion_states_no_annotation_nor_gen_total += 1
                    
                    last_is_assert = last_is_assert_tmp
                    i += 1
                    continue
                else:
                    # current essential state matches UIState in the exec trace,
                    # go to the next essential state in the ground-truth trace
                    # and the next UIState in the exec trace
                    tmp_pages_annotated_total += 1
                    gr_ui_state_matched = True
                    self.completed_key_subtasks += 1
                    
                    # Track action position for finished key subtask
                    self.total_actions_pos_for_finished_key_subtask += i_no_oracle
                    
                    if difficulty == "easy":
                        self.completed_key_subtasks_easy += 1
                        self.total_actions_pos_for_finished_key_subtask_easy += i_no_oracle
                    elif difficulty == "mid":
                        self.completed_key_subtasks_mid += 1
                        self.total_actions_pos_for_finished_key_subtask_mid += i_no_oracle
                    else:
                        self.completed_key_subtasks_hard += 1
                        self.total_actions_pos_for_finished_key_subtask_hard += i_no_oracle
                    # click_match_states: List[str] = ui_state.get(EssentialStateKeyword.CLICK, None)
                    i,  hit_metrics = self.calc_hit_results(episode, ui_state, exec_trace, i) # # 114 (without), 100? with this line ; 排查 Oracle.view is None 的原因
                    last_is_assert = False # 自动向前一步，last_is_assert = False 
                        
                    self.pages_exec_total += tmp_pages_exec_total
                    self.pages_gen_total += tmp_pages_gen_total
                    self.pages_annotated_total += tmp_pages_annotated_total
                    self.pages_annotated_hit += tmp_pages_annotated_hit + hit_metrics.pages_annotated_hit # only latter added
                    self.pages_no_annotation_nor_gen_total += tmp_pages_no_annotation_nor_gen_total
                    
                    # Track page-level metrics by difficulty
                    if difficulty == "easy":
                        self.pages_exec_total_easy += tmp_pages_exec_total
                        self.pages_gen_total_easy += tmp_pages_gen_total
                        self.pages_annotated_total_easy += tmp_pages_annotated_total
                        self.pages_annotated_hit_easy += tmp_pages_annotated_hit + hit_metrics.pages_annotated_hit
                        self.pages_no_annotation_nor_gen_total_easy += tmp_pages_no_annotation_nor_gen_total
                    elif difficulty == "mid":
                        self.pages_exec_total_mid += tmp_pages_exec_total
                        self.pages_gen_total_mid += tmp_pages_gen_total
                        self.pages_annotated_total_mid += tmp_pages_annotated_total
                        self.pages_annotated_hit_mid += tmp_pages_annotated_hit + hit_metrics.pages_annotated_hit
                        self.pages_no_annotation_nor_gen_total_mid += tmp_pages_no_annotation_nor_gen_total
                    else:
                        self.pages_exec_total_hard += tmp_pages_exec_total
                        self.pages_gen_total_hard += tmp_pages_gen_total
                        self.pages_annotated_total_hard += tmp_pages_annotated_total
                        self.pages_annotated_hit_hard += tmp_pages_annotated_hit + hit_metrics.pages_annotated_hit
                        self.pages_no_annotation_nor_gen_total_hard += tmp_pages_no_annotation_nor_gen_total
                    
                    self.assertion_states_exec_total += tmp_assertion_states_exec_total + hit_metrics.assertion_states_exec_total
                    self.assertion_states_gen_total += tmp_assertion_states_gen_total + hit_metrics.assertion_states_gen_total
                    self.assertion_states_annotated_total += tmp_assertion_states_annotated_total + hit_metrics.assertion_states_annotated_total # only latter added
                    self.assertion_states_annotated_hit += tmp_assertion_states_annotated_hit + hit_metrics.assertion_states_annotated_hit # only latter added
                    self.assertion_states_no_annotation_nor_gen_total += tmp_assertion_states_no_annotation_nor_gen_total
                    
                    # Track recognized pages with widget assertions by difficulty
                    if hit_metrics.is_assert_widget_set and hit_metrics.pages_annotated_hit > 0: # only count when page has widget assertions and assertion is generated
                        self.recognized_pages_with_widget_assertions += 1
                        if difficulty == "easy":
                            self.recognized_pages_with_widget_assertions_easy += 1
                        elif difficulty == "mid":
                            self.recognized_pages_with_widget_assertions_mid += 1
                        else:
                            self.recognized_pages_with_widget_assertions_hard += 1
                        
                        if hit_metrics.assertion_states_annotated_hit > 0:
                            self.recognized_pages_with_widget_assertions_and_evidence_widget_hits += 1
                            if difficulty == "easy":
                                self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy += 1
                            elif difficulty == "mid":
                                self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid += 1
                            else:
                                self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard += 1
                                            
                    tmp_pages_exec_total: int = 0 # total pages in execution trace, denominator(分母) for page-level accuracy
                    tmp_pages_gen_total: int = 0 # total pages with generated assertions, denominator(分母) for page-level precision
                    tmp_pages_annotated_total: int = 0 # total pages with annotated assertions, denominator(分母) for page-level recall
                    tmp_pages_annotated_hit: int = 0 # pages with annotated assertions and at least one hit, numerator(分子) for page-level recall and precision
                    tmp_pages_no_annotation_nor_gen_total: int = 0 # true negative: pages without annotated assertions and without generated assertions; along with pages_annotated_hit, numerator(分子) for page-level accuracy
                    # assert here
                    # 验证混淆矩阵一致性
                    TP_pages = self.pages_annotated_hit
                    FP_pages = self.pages_gen_total - self.pages_annotated_hit
                    FN_pages = self.pages_annotated_total - self.pages_annotated_hit
                    TN_pages = self.pages_no_annotation_nor_gen_total
                    total_calculated = TP_pages + FP_pages + FN_pages + TN_pages
                    
                    if total_calculated != self.pages_exec_total:
                        error_log_file = f"dumped_stats/confusion_matrix_errors_{self.agent.agent_name}_limit_{self.no_oracle_step_limit}.log"
                        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
                        with open(error_log_file, "a", encoding="utf-8") as f:
                            f.write(f"Episode: {episode}\n")
                            f.write(f"  pages_exec_total = {self.pages_exec_total}\n")
                            f.write(f"  pages_gen_total = {self.pages_gen_total}\n")
                            f.write(f"  pages_annotated_total = {self.pages_annotated_total}\n")
                            f.write(f"  pages_annotated_hit = {self.pages_annotated_hit}\n")
                            f.write(f"  pages_no_annotation_nor_gen_total = {self.pages_no_annotation_nor_gen_total}\n")
                            f.write(f"  TP_pages = {TP_pages}\n")
                            f.write(f"  FP_pages = {FP_pages}\n")
                            f.write(f"  FN_pages = {FN_pages}\n")
                            f.write(f"  TN_pages = {TN_pages}\n")
                            f.write(f"  TP + FP + FN + TN = {total_calculated} != pages_exec_total = {self.pages_exec_total}\n")
                            f.write(f"  Difference = {total_calculated - self.pages_exec_total}\n")
                            f.write("-" * 50 + "\n")
                    
                    tmp_assertion_states_exec_total: int = 0 # total assertion states in execution trace, denominator(分母) for assertion-level accuracy
                    tmp_assertion_states_gen_total: int = 0 # total generated assertion states, denominator(分母) for assertion-level precision
                    tmp_assertion_states_annotated_total: int = 0 # total annotated assertion states, denominator(分母) for assertion-level recall
                    tmp_assertion_states_annotated_hit: int = 0 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
                    tmp_assertion_states_no_annotation_nor_gen_total: int = 0 # true negative: assertion states without annotated assertions and without generated assertions;
                    
                    
                    break

            if gr_ui_state_matched:
                continue
            else:
                return False, None

        # Task completed successfully - track completed episodes by difficulty
        self.completed_episodes_count += 1
        if difficulty == "easy":
            self.completed_episodes_count_easy += 1
        elif difficulty == "mid":
            self.completed_episodes_count_mid += 1
        else:
            self.completed_episodes_count_hard += 1

        return True, None

    def post_evaluation_hook(self) -> None:
        """
        Dump hit results after evaluation.
        """
        print(f"DEBUG: hit_results length: {len(self.hit_results)}")  # 添加调试信息
        if self.hit_results:
            print(f"DEBUG: First hit result: {self.hit_results[0]}")

        file_name = f"dumped_stats_final/oracle_hit_results_{self.evaluator_name}_{self.agent.agent_name}_limit_{self.no_oracle_step_limit}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv"
        self.dump_hit_results(file_name)
        
        # 计算并输出统计指标
        metrics_file_name = f"dumped_stats_final/evaluation_metrics_{self.evaluator_name}_{self.agent.agent_name}_limit_{self.no_oracle_step_limit}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv"
        self.dump_evaluation_metrics(metrics_file_name)
        

    def dump_hit_results(self, filename: str = "oracle_hit_results.csv"):
        import csv
        import os
        
        print(f"DEBUG: Trying to dump {len(self.hit_results)} results to {filename}")  # 添加调试信息
        if not self.hit_results:
            print("DEBUG: hit_results is empty, not creating file")  # 添加调试信息
            return
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 获取所有字段名，并添加统计字段
        hit_results_fieldnames = list(self.hit_results[0]._fields)
        
        # 添加统计变量字段
        stats_fieldnames = [
            "key_subtasks_total", "completed_key_subtasks",
            "pages_exec_total", "pages_gen_total", "pages_annotated_total", 
            "pages_annotated_hit", "pages_no_annotation_nor_gen_total",
            "assertion_states_exec_total", "assertion_states_gen_total", 
            "assertion_states_annotated_total", "assertion_states_annotated_hit", 
            "assertion_states_no_annotation_nor_gen_total",
            # 计算得出的指标
            "page_precision", "page_recall", "page_accuracy", "page_f1",
            "assertion_precision", "assertion_recall", "assertion_accuracy", "assertion_f1",
            "task_completion_rate"
        ]
        
        all_fieldnames = hit_results_fieldnames + stats_fieldnames
        
        # 计算指标
        page_precision = self.pages_annotated_hit / self.pages_gen_total if self.pages_gen_total > 0 else 0.0
        page_recall = self.pages_annotated_hit / self.pages_annotated_total if self.pages_annotated_total > 0 else 0.0
        page_accuracy = (self.pages_annotated_hit + self.pages_no_annotation_nor_gen_total) / self.pages_exec_total if self.pages_exec_total > 0 else 0.0
        page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if (page_precision + page_recall) > 0 else 0.0
        
        assertion_precision = self.assertion_states_annotated_hit / self.assertion_states_gen_total if self.assertion_states_gen_total > 0 else 0.0
        assertion_recall = self.assertion_states_annotated_hit / self.assertion_states_annotated_total if self.assertion_states_annotated_total > 0 else 0.0
        assertion_accuracy = (self.assertion_states_annotated_hit + self.assertion_states_no_annotation_nor_gen_total) / self.assertion_states_exec_total if self.assertion_states_exec_total > 0 else 0.0
        assertion_f1 = 2 * assertion_precision * assertion_recall / (assertion_precision + assertion_recall) if (assertion_precision + assertion_recall) > 0 else 0.0
        
        task_completion_rate = self.completed_key_subtasks / self.key_subtasks_total if self.key_subtasks_total > 0 else 0.0
        
        with open(filename, "w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            for tup in self.hit_results:
                # NamedTuple转dict
                row = tup._asdict()
                # 列表字段转字符串
                for k, v in row.items():
                    if isinstance(v, list):
                        row[k] = ",".join(str(x) for x in v)
                
                # 添加统计变量
                row.update({
                    "key_subtasks_total": self.key_subtasks_total,
                    "completed_key_subtasks": self.completed_key_subtasks,
                    "pages_exec_total": self.pages_exec_total,
                    "pages_gen_total": self.pages_gen_total,
                    "pages_annotated_total": self.pages_annotated_total,
                    "pages_annotated_hit": self.pages_annotated_hit,
                    "pages_no_annotation_nor_gen_total": self.pages_no_annotation_nor_gen_total,
                    "assertion_states_exec_total": self.assertion_states_exec_total,
                    "assertion_states_gen_total": self.assertion_states_gen_total,
                    "assertion_states_annotated_total": self.assertion_states_annotated_total,
                    "assertion_states_annotated_hit": self.assertion_states_annotated_hit,
                    "assertion_states_no_annotation_nor_gen_total": self.assertion_states_no_annotation_nor_gen_total,
                    # 计算得出的指标
                    "page_precision": round(page_precision, 4),
                    "page_recall": round(page_recall, 4),
                    "page_accuracy": round(page_accuracy, 4),
                    "page_f1": round(page_f1, 4),
                    "assertion_precision": round(assertion_precision, 4),
                    "assertion_recall": round(assertion_recall, 4),
                    "assertion_accuracy": round(assertion_accuracy, 4),
                    "assertion_f1": round(assertion_f1, 4),
                    "task_completion_rate": round(task_completion_rate, 4),
                })
                
                writer.writerow(row)
    
    def dump_evaluation_metrics(self, filename: str = "evaluation_metrics.csv"):
        """
        计算并输出页面级别和断言级别的recall, precision, accuracy等指标
        """
        import csv
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # 计算页面级别指标 (Overall)
        page_precision = self.pages_annotated_hit / self.pages_gen_total if self.pages_gen_total > 0 else 0.0
        page_recall = self.pages_annotated_hit / self.pages_annotated_total if self.pages_annotated_total > 0 else 0.0
        page_accuracy = (self.pages_annotated_hit + self.pages_no_annotation_nor_gen_total) / self.pages_exec_total if self.pages_exec_total > 0 else 0.0
        page_f1 = 2 * page_precision * page_recall / (page_precision + page_recall) if (page_precision + page_recall) > 0 else 0.0
        
        # 计算页面级别指标 (Easy)
        page_precision_easy = self.pages_annotated_hit_easy / self.pages_gen_total_easy if self.pages_gen_total_easy > 0 else 0.0
        page_recall_easy = self.pages_annotated_hit_easy / self.pages_annotated_total_easy if self.pages_annotated_total_easy > 0 else 0.0
        page_accuracy_easy = (self.pages_annotated_hit_easy + self.pages_no_annotation_nor_gen_total_easy) / self.pages_exec_total_easy if self.pages_exec_total_easy > 0 else 0.0
        page_f1_easy = 2 * page_precision_easy * page_recall_easy / (page_precision_easy + page_recall_easy) if (page_precision_easy + page_recall_easy) > 0 else 0.0
        
        # 计算页面级别指标 (Mid)
        page_precision_mid = self.pages_annotated_hit_mid / self.pages_gen_total_mid if self.pages_gen_total_mid > 0 else 0.0
        page_recall_mid = self.pages_annotated_hit_mid / self.pages_annotated_total_mid if self.pages_annotated_total_mid > 0 else 0.0
        page_accuracy_mid = (self.pages_annotated_hit_mid + self.pages_no_annotation_nor_gen_total_mid) / self.pages_exec_total_mid if self.pages_exec_total_mid > 0 else 0.0
        page_f1_mid = 2 * page_precision_mid * page_recall_mid / (page_precision_mid + page_recall_mid) if (page_precision_mid + page_recall_mid) > 0 else 0.0
        
        # 计算页面级别指标 (Hard)
        page_precision_hard = self.pages_annotated_hit_hard / self.pages_gen_total_hard if self.pages_gen_total_hard > 0 else 0.0
        page_recall_hard = self.pages_annotated_hit_hard / self.pages_annotated_total_hard if self.pages_annotated_total_hard > 0 else 0.0
        page_accuracy_hard = (self.pages_annotated_hit_hard + self.pages_no_annotation_nor_gen_total_hard) / self.pages_exec_total_hard if self.pages_exec_total_hard > 0 else 0.0
        page_f1_hard = 2 * page_precision_hard * page_recall_hard / (page_precision_hard + page_recall_hard) if (page_precision_hard + page_recall_hard) > 0 else 0.0
        
        # 计算断言级别指标 (Overall)
        assertion_precision = self.assertion_states_annotated_hit / self.assertion_states_gen_total if self.assertion_states_gen_total > 0 else 0.0
        assertion_recall = self.assertion_states_annotated_hit / self.assertion_states_annotated_total if self.assertion_states_annotated_total > 0 else 0.0
        assertion_accuracy = (self.assertion_states_annotated_hit + self.assertion_states_no_annotation_nor_gen_total) / self.assertion_states_exec_total if self.assertion_states_exec_total > 0 else 0.0
        assertion_f1 = 2 * assertion_precision * assertion_recall / (assertion_precision + assertion_recall) if (assertion_precision + assertion_recall) > 0 else 0.0
        
        # 计算TAR (Task Completion Rate)
        self.TAR = self.completed_episodes_count / self.unerror_episodes_total if self.unerror_episodes_total > 0 else 0.0
        self.TAR_easy = self.completed_episodes_count_easy / self.unerror_episodes_total_easy if self.unerror_episodes_total_easy > 0 else 0.0
        self.TAR_mid = self.completed_episodes_count_mid / self.unerror_episodes_total_mid if self.unerror_episodes_total_mid > 0 else 0.0
        self.TAR_hard = self.completed_episodes_count_hard / self.unerror_episodes_total_hard if self.unerror_episodes_total_hard > 0 else 0.0
        
        # 计算KSAR (Key Subtask Completion Rate)
        self.KSAR = self.completed_key_subtasks / self.key_subtasks_total if self.key_subtasks_total > 0 else 0.0
        self.KSAR_easy = self.completed_key_subtasks_easy / self.key_subtasks_total_easy if self.key_subtasks_total_easy > 0 else 0.0
        self.KSAR_mid = self.completed_key_subtasks_mid / self.key_subtasks_total_mid if self.key_subtasks_total_mid > 0 else 0.0
        self.KSAR_hard = self.completed_key_subtasks_hard / self.key_subtasks_total_hard if self.key_subtasks_total_hard > 0 else 0.0
        
        # 计算widget_hit_rate
        self.widget_hit_rate = self.recognized_pages_with_widget_assertions_and_evidence_widget_hits / self.recognized_pages_with_widget_assertions if self.recognized_pages_with_widget_assertions > 0 else 0.0
        self.widget_hit_rate_easy = self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy / self.recognized_pages_with_widget_assertions_easy if self.recognized_pages_with_widget_assertions_easy > 0 else 0.0
        self.widget_hit_rate_mid = self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid / self.recognized_pages_with_widget_assertions_mid if self.recognized_pages_with_widget_assertions_mid > 0 else 0.0
        self.widget_hit_rate_hard = self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard / self.recognized_pages_with_widget_assertions_hard if self.recognized_pages_with_widget_assertions_hard > 0 else 0.0
        
        # 计算average_actions_pos_for_finished_key_subtask
        self.average_actions_pos_for_finished_key_subtask = self.total_actions_pos_for_finished_key_subtask / self.completed_key_subtasks if self.completed_key_subtasks > 0 else 0.0
        self.average_actions_pos_for_finished_key_subtask_easy = self.total_actions_pos_for_finished_key_subtask_easy / self.completed_key_subtasks_easy if self.completed_key_subtasks_easy > 0 else 0.0
        self.average_actions_pos_for_finished_key_subtask_mid = self.total_actions_pos_for_finished_key_subtask_mid / self.completed_key_subtasks_mid if self.completed_key_subtasks_mid > 0 else 0.0
        self.average_actions_pos_for_finished_key_subtask_hard = self.total_actions_pos_for_finished_key_subtask_hard / self.completed_key_subtasks_hard if self.completed_key_subtasks_hard > 0 else 0.0
        
        # 准备输出数据
        metrics_data = [
            # ========== Overall Metrics ==========
            # TAR (Task Completion Rate)
            {"metric_type": "task_overall", "metric_name": "TAR", "value": self.TAR,
             "numerator": self.completed_episodes_count, "denominator": self.unerror_episodes_total},
            
            # KSAR (Key Subtask Completion Rate)
            {"metric_type": "task_overall", "metric_name": "KSAR", "value": self.KSAR,
             "numerator": self.completed_key_subtasks, "denominator": self.key_subtasks_total},
            
            # Widget Hit Rate
            {"metric_type": "widget_overall", "metric_name": "widget_hit_rate", "value": self.widget_hit_rate,
             "numerator": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits, "denominator": self.recognized_pages_with_widget_assertions},
            
            # Average Actions Position
            {"metric_type": "efficiency_overall", "metric_name": "avg_actions_pos", "value": self.average_actions_pos_for_finished_key_subtask,
             "numerator": self.total_actions_pos_for_finished_key_subtask, "denominator": self.completed_key_subtasks},
            
            # 页面级别指标
            {"metric_type": "page_overall", "metric_name": "precision", "value": page_precision,
             "numerator": self.pages_annotated_hit, "denominator": self.pages_gen_total},
            {"metric_type": "page_overall", "metric_name": "recall", "value": page_recall,
             "numerator": self.pages_annotated_hit, "denominator": self.pages_annotated_total},
            {"metric_type": "page_overall", "metric_name": "accuracy", "value": page_accuracy,
             "numerator": self.pages_annotated_hit + self.pages_no_annotation_nor_gen_total, "denominator": self.pages_exec_total},
            {"metric_type": "page_overall", "metric_name": "f1_score", "value": page_f1,
             "numerator": "2*P*R/(P+R)", "denominator": "calculated"},
            
            # 断言级别指标
            {"metric_type": "assertion_overall", "metric_name": "precision", "value": assertion_precision,
             "numerator": self.assertion_states_annotated_hit, "denominator": self.assertion_states_gen_total},
            {"metric_type": "assertion_overall", "metric_name": "recall", "value": assertion_recall,
             "numerator": self.assertion_states_annotated_hit, "denominator": self.assertion_states_annotated_total},
            {"metric_type": "assertion_overall", "metric_name": "accuracy", "value": assertion_accuracy,
             "numerator": self.assertion_states_annotated_hit + self.assertion_states_no_annotation_nor_gen_total, "denominator": self.assertion_states_exec_total},
            {"metric_type": "assertion_overall", "metric_name": "f1_score", "value": assertion_f1,
             "numerator": "2*P*R/(P+R)", "denominator": "calculated"},
            
            # ========== Easy Difficulty Metrics ==========
            {"metric_type": "task_easy", "metric_name": "TAR", "value": self.TAR_easy,
             "numerator": self.completed_episodes_count_easy, "denominator": self.unerror_episodes_total_easy},
            {"metric_type": "task_easy", "metric_name": "KSAR", "value": self.KSAR_easy,
             "numerator": self.completed_key_subtasks_easy, "denominator": self.key_subtasks_total_easy},
            {"metric_type": "widget_easy", "metric_name": "widget_hit_rate", "value": self.widget_hit_rate_easy,
             "numerator": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy, "denominator": self.recognized_pages_with_widget_assertions_easy},
            {"metric_type": "efficiency_easy", "metric_name": "avg_actions_pos", "value": self.average_actions_pos_for_finished_key_subtask_easy,
             "numerator": self.total_actions_pos_for_finished_key_subtask_easy, "denominator": self.completed_key_subtasks_easy},
            {"metric_type": "page_easy", "metric_name": "precision", "value": page_precision_easy,
             "numerator": self.pages_annotated_hit_easy, "denominator": self.pages_gen_total_easy},
            {"metric_type": "page_easy", "metric_name": "recall", "value": page_recall_easy,
             "numerator": self.pages_annotated_hit_easy, "denominator": self.pages_annotated_total_easy},
            {"metric_type": "page_easy", "metric_name": "accuracy", "value": page_accuracy_easy,
             "numerator": self.pages_annotated_hit_easy + self.pages_no_annotation_nor_gen_total_easy, "denominator": self.pages_exec_total_easy},
            {"metric_type": "page_easy", "metric_name": "f1_score", "value": page_f1_easy,
             "numerator": "2*P*R/(P+R)", "denominator": "calculated"},
            
            # ========== Mid Difficulty Metrics ==========
            {"metric_type": "task_mid", "metric_name": "TAR", "value": self.TAR_mid,
             "numerator": self.completed_episodes_count_mid, "denominator": self.unerror_episodes_total_mid},
            {"metric_type": "task_mid", "metric_name": "KSAR", "value": self.KSAR_mid,
             "numerator": self.completed_key_subtasks_mid, "denominator": self.key_subtasks_total_mid},
            {"metric_type": "widget_mid", "metric_name": "widget_hit_rate", "value": self.widget_hit_rate_mid,
             "numerator": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid, "denominator": self.recognized_pages_with_widget_assertions_mid},
            {"metric_type": "efficiency_mid", "metric_name": "avg_actions_pos", "value": self.average_actions_pos_for_finished_key_subtask_mid,
             "numerator": self.total_actions_pos_for_finished_key_subtask_mid, "denominator": self.completed_key_subtasks_mid},
            {"metric_type": "page_mid", "metric_name": "precision", "value": page_precision_mid,
             "numerator": self.pages_annotated_hit_mid, "denominator": self.pages_gen_total_mid},
            {"metric_type": "page_mid", "metric_name": "recall", "value": page_recall_mid,
             "numerator": self.pages_annotated_hit_mid, "denominator": self.pages_annotated_total_mid},
            {"metric_type": "page_mid", "metric_name": "accuracy", "value": page_accuracy_mid,
             "numerator": self.pages_annotated_hit_mid + self.pages_no_annotation_nor_gen_total_mid, "denominator": self.pages_exec_total_mid},
            {"metric_type": "page_mid", "metric_name": "f1_score", "value": page_f1_mid,
             "numerator": "2*P*R/(P+R)", "denominator": "calculated"},
            
            # ========== Hard Difficulty Metrics ==========
            {"metric_type": "task_hard", "metric_name": "TAR", "value": self.TAR_hard,
             "numerator": self.completed_episodes_count_hard, "denominator": self.unerror_episodes_total_hard},
            {"metric_type": "task_hard", "metric_name": "KSAR", "value": self.KSAR_hard,
             "numerator": self.completed_key_subtasks_hard, "denominator": self.key_subtasks_total_hard},
            {"metric_type": "widget_hard", "metric_name": "widget_hit_rate", "value": self.widget_hit_rate_hard,
             "numerator": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard, "denominator": self.recognized_pages_with_widget_assertions_hard},
            {"metric_type": "efficiency_hard", "metric_name": "avg_actions_pos", "value": self.average_actions_pos_for_finished_key_subtask_hard,
             "numerator": self.total_actions_pos_for_finished_key_subtask_hard, "denominator": self.completed_key_subtasks_hard},
            {"metric_type": "page_hard", "metric_name": "precision", "value": page_precision_hard,
             "numerator": self.pages_annotated_hit_hard, "denominator": self.pages_gen_total_hard},
            {"metric_type": "page_hard", "metric_name": "recall", "value": page_recall_hard,
             "numerator": self.pages_annotated_hit_hard, "denominator": self.pages_annotated_total_hard},
            {"metric_type": "page_hard", "metric_name": "accuracy", "value": page_accuracy_hard,
             "numerator": self.pages_annotated_hit_hard + self.pages_no_annotation_nor_gen_total_hard, "denominator": self.pages_exec_total_hard},
            {"metric_type": "page_hard", "metric_name": "f1_score", "value": page_f1_hard,
             "numerator": "2*P*R/(P+R)", "denominator": "calculated"},
        ]
        
        # 添加原始统计数据
        raw_stats = [
            # Overall raw stats
            {"metric_type": "raw_stats_overall", "metric_name": "unerror_episodes_total", "value": self.unerror_episodes_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "completed_episodes_count", "value": self.completed_episodes_count, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "key_subtasks_total", "value": self.key_subtasks_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "completed_key_subtasks", "value": self.completed_key_subtasks, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "pages_exec_total", "value": self.pages_exec_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "pages_gen_total", "value": self.pages_gen_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "pages_annotated_total", "value": self.pages_annotated_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "pages_annotated_hit", "value": self.pages_annotated_hit, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "pages_no_annotation_nor_gen_total", "value": self.pages_no_annotation_nor_gen_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "assertion_states_exec_total", "value": self.assertion_states_exec_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "assertion_states_gen_total", "value": self.assertion_states_gen_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "assertion_states_annotated_total", "value": self.assertion_states_annotated_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "assertion_states_annotated_hit", "value": self.assertion_states_annotated_hit, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "assertion_states_no_annotation_nor_gen_total", "value": self.assertion_states_no_annotation_nor_gen_total, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "recognized_pages_with_widget_assertions", "value": self.recognized_pages_with_widget_assertions, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "recognized_pages_with_widget_assertions_and_evidence_widget_hits", "value": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_overall", "metric_name": "total_actions_pos_for_finished_key_subtask", "value": self.total_actions_pos_for_finished_key_subtask, "numerator": "", "denominator": ""},
            
            # Easy raw stats
            {"metric_type": "raw_stats_easy", "metric_name": "unerror_episodes_total", "value": self.unerror_episodes_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "completed_episodes_count", "value": self.completed_episodes_count_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "key_subtasks_total", "value": self.key_subtasks_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "completed_key_subtasks", "value": self.completed_key_subtasks_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "pages_exec_total", "value": self.pages_exec_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "pages_gen_total", "value": self.pages_gen_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "pages_annotated_total", "value": self.pages_annotated_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "pages_annotated_hit", "value": self.pages_annotated_hit_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "pages_no_annotation_nor_gen_total", "value": self.pages_no_annotation_nor_gen_total_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "recognized_pages_with_widget_assertions", "value": self.recognized_pages_with_widget_assertions_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "recognized_pages_with_widget_assertions_and_evidence_widget_hits", "value": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_easy", "metric_name": "total_actions_pos_for_finished_key_subtask", "value": self.total_actions_pos_for_finished_key_subtask_easy, "numerator": "", "denominator": ""},
            
            # Mid raw stats
            {"metric_type": "raw_stats_mid", "metric_name": "unerror_episodes_total", "value": self.unerror_episodes_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "completed_episodes_count", "value": self.completed_episodes_count_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "key_subtasks_total", "value": self.key_subtasks_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "completed_key_subtasks", "value": self.completed_key_subtasks_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "pages_exec_total", "value": self.pages_exec_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "pages_gen_total", "value": self.pages_gen_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "pages_annotated_total", "value": self.pages_annotated_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "pages_annotated_hit", "value": self.pages_annotated_hit_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "pages_no_annotation_nor_gen_total", "value": self.pages_no_annotation_nor_gen_total_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "recognized_pages_with_widget_assertions", "value": self.recognized_pages_with_widget_assertions_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "recognized_pages_with_widget_assertions_and_evidence_widget_hits", "value": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_mid", "metric_name": "total_actions_pos_for_finished_key_subtask", "value": self.total_actions_pos_for_finished_key_subtask_mid, "numerator": "", "denominator": ""},
            
            # Hard raw stats
            {"metric_type": "raw_stats_hard", "metric_name": "unerror_episodes_total", "value": self.unerror_episodes_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "completed_episodes_count", "value": self.completed_episodes_count_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "key_subtasks_total", "value": self.key_subtasks_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "completed_key_subtasks", "value": self.completed_key_subtasks_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "pages_exec_total", "value": self.pages_exec_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "pages_gen_total", "value": self.pages_gen_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "pages_annotated_total", "value": self.pages_annotated_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "pages_annotated_hit", "value": self.pages_annotated_hit_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "pages_no_annotation_nor_gen_total", "value": self.pages_no_annotation_nor_gen_total_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "recognized_pages_with_widget_assertions", "value": self.recognized_pages_with_widget_assertions_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "recognized_pages_with_widget_assertions_and_evidence_widget_hits", "value": self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard, "numerator": "", "denominator": ""},
            {"metric_type": "raw_stats_hard", "metric_name": "total_actions_pos_for_finished_key_subtask", "value": self.total_actions_pos_for_finished_key_subtask_hard, "numerator": "", "denominator": ""},
        ]
        
        # 合并所有数据
        all_data = metrics_data + raw_stats
        
        # 写入CSV文件
        with open(filename, "w", newline='', encoding="utf-8") as f:
            fieldnames = ["metric_type", "metric_name", "value", "numerator", "denominator"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
        
        # 打印关键指标到控制台
        print(f"\n{'='*60}")
        print(f"=== Evaluation Metrics (Overall) ===")
        print(f"{'='*60}")
        print(f"TAR (Task Completion Rate): {self.TAR:.4f} ({self.completed_episodes_count}/{self.unerror_episodes_total})")
        print(f"KSAR (Key Subtask Completion Rate): {self.KSAR:.4f} ({self.completed_key_subtasks}/{self.key_subtasks_total})")
        print(f"Widget Hit Rate: {self.widget_hit_rate:.4f} ({self.recognized_pages_with_widget_assertions_and_evidence_widget_hits}/{self.recognized_pages_with_widget_assertions})")
        print(f"Avg Actions Pos for Finished Key Subtask: {self.average_actions_pos_for_finished_key_subtask:.4f}")
        print(f"\nPage-level Metrics:")
        print(f"  Precision: {page_precision:.4f} ({self.pages_annotated_hit}/{self.pages_gen_total})")
        print(f"  Recall:    {page_recall:.4f} ({self.pages_annotated_hit}/{self.pages_annotated_total})")
        print(f"  Accuracy:  {page_accuracy:.4f} ({self.pages_annotated_hit + self.pages_no_annotation_nor_gen_total}/{self.pages_exec_total})")
        print(f"  F1-Score:  {page_f1:.4f}")
        print(f"\nAssertion-level Metrics:")
        print(f"  Precision: {assertion_precision:.4f} ({self.assertion_states_annotated_hit}/{self.assertion_states_gen_total})")
        print(f"  Recall:    {assertion_recall:.4f} ({self.assertion_states_annotated_hit}/{self.assertion_states_annotated_total})")
        print(f"  Accuracy:  {assertion_accuracy:.4f} ({self.assertion_states_annotated_hit + self.assertion_states_no_annotation_nor_gen_total}/{self.assertion_states_exec_total})")
        print(f"  F1-Score:  {assertion_f1:.4f}")
        
        print(f"\n{'='*60}")
        print(f"=== Metrics by Difficulty ===")
        print(f"{'='*60}")
        print(f"\n--- Easy (gr_trace_len < 4) ---")
        print(f"TAR: {self.TAR_easy:.4f} ({self.completed_episodes_count_easy}/{self.unerror_episodes_total_easy})")
        print(f"KSAR: {self.KSAR_easy:.4f} ({self.completed_key_subtasks_easy}/{self.key_subtasks_total_easy})")
        print(f"Widget Hit Rate: {self.widget_hit_rate_easy:.4f} ({self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_easy}/{self.recognized_pages_with_widget_assertions_easy})")
        print(f"Avg Actions Pos: {self.average_actions_pos_for_finished_key_subtask_easy:.4f}")
        print(f"Page P/R/A/F1: {page_precision_easy:.4f}/{page_recall_easy:.4f}/{page_accuracy_easy:.4f}/{page_f1_easy:.4f}")
        
        print(f"\n--- Mid (4 <= gr_trace_len < 8) ---")
        print(f"TAR: {self.TAR_mid:.4f} ({self.completed_episodes_count_mid}/{self.unerror_episodes_total_mid})")
        print(f"KSAR: {self.KSAR_mid:.4f} ({self.completed_key_subtasks_mid}/{self.key_subtasks_total_mid})")
        print(f"Widget Hit Rate: {self.widget_hit_rate_mid:.4f} ({self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_mid}/{self.recognized_pages_with_widget_assertions_mid})")
        print(f"Avg Actions Pos: {self.average_actions_pos_for_finished_key_subtask_mid:.4f}")
        print(f"Page P/R/A/F1: {page_precision_mid:.4f}/{page_recall_mid:.4f}/{page_accuracy_mid:.4f}/{page_f1_mid:.4f}")
        
        print(f"\n--- Hard (gr_trace_len >= 8) ---")
        print(f"TAR: {self.TAR_hard:.4f} ({self.completed_episodes_count_hard}/{self.unerror_episodes_total_hard})")
        print(f"KSAR: {self.KSAR_hard:.4f} ({self.completed_key_subtasks_hard}/{self.key_subtasks_total_hard})")
        print(f"Widget Hit Rate: {self.widget_hit_rate_hard:.4f} ({self.recognized_pages_with_widget_assertions_and_evidence_widget_hits_hard}/{self.recognized_pages_with_widget_assertions_hard})")
        print(f"Avg Actions Pos: {self.average_actions_pos_for_finished_key_subtask_hard:.4f}")
        print(f"Page P/R/A/F1: {page_precision_hard:.4f}/{page_recall_hard:.4f}/{page_accuracy_hard:.4f}/{page_f1_hard:.4f}")
        
        print(f"\nMetrics saved to: {filename}")
        
    # 69947946018315292528: 图片和VH有差别，但是img，VH都匹配; 另外uicomponent应该匹配但是没匹配
    
    def calc_hit_results(
        self, episode: str, gr_ui_state: UIState, exec_trace: TaskTrace, node_id: int
    ) -> int:
        """
        计算当前ground-truth UIState的命中结果，并更新hit_results列表
        
        Args:
            episode: 当前episode标识符
            gr_ui_state: ground-truth UI状态
            exec_trace: 执行轨迹
            node_id: 执行轨迹中的起始节点ID
            
        Returns:
            下一个要处理的节点ID
        """
        # 初始化各种命中指标
        page_hit: int = 0  # 页面匹配命中
        activity_hit: int = 0  # 基于activity判定的页面匹配命中
        fuzzy_pageimg_hit: int = 0  # 页面图片模糊匹配命中
        fuzzy_page_main_uicomps_hit: int = 0  # 页面主要UI组件模糊匹配命中
        uicomponent_nearFull_hit: int = 0  # UI组件近似完全匹配命中（等同于精确匹配）
        uicomponent_keywords_hit: int = 0  # UI组件关键词匹配命中
        uicomponent_text_hit: int = 0  # UI组件文本匹配命中
        uicomponent_set: int = 0  # 是否设置了UI组件断言：{exact<i>, fuzzy<i>, i>=0} 
        
        # 记录匹配命中的具体节点ID列表
        uicomp_nearFull_hit_list: List[int] = []
        uicomp_keywords_hit_list: List[int] = []
        uicomp_text_hit_list: List[int] = []
        

        tmp_pages_annotated_hit: int = 0 # pages with annotated assertions and at least one hit, numerator(分子) for page-level recall and precision

        tmp_assertion_states_exec_total: int = 0 # total assertion states in execution trace, denominator(分母) for assertion-level accuracy
        tmp_assertion_states_gen_total: int = 0 # total generated assertion states, denominator(分母) for assertion-level precision               
        tmp_assertion_states_annotated_total: int = 0 # total annotated assertion states, denominator(分母) for assertion-level recall
        tmp_assertion_states_annotated_hit: int = 0 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
        

        # 获取ground-truth的essential state字典
        es_dict = gr_ui_state.essential_state
        first_cmp = True  # 标记是否是第一次比较
        node_start_id = node_id #记录gr_ui_state匹配开始的节点ID
        node_start_oracle_id = -1  # 记录Oracle事件开始的节点ID
        
        exec_ui_state_first = exec_trace[node_id]
        exec_ui_state = exec_trace[node_id]

        # 【优化1】预先构建匹配过滤器，避免在循环中重复构建
        match_filter: List[int] = []
        
        # 处理exact匹配状态
        uicomponent_match_states: List[str] = es_dict.get(EssentialStateKeyword.EXACT, None)
        if uicomponent_match_states:
            match_filter.extend([int(node_id_str) for node_id_str in uicomponent_match_states])
        
        # 处理fuzzy匹配状态
        fuzzy_match_states: List[str] = es_dict.get(EssentialStateKeyword.FUZZY, None)
        if fuzzy_match_states:
            # 只添加有效的节点ID（>=0）
            valid_fuzzy_ids = [int(node_id_str) for node_id_str in fuzzy_match_states if int(node_id_str) >= 0]
            match_filter.extend(valid_fuzzy_ids)
            
        # 处理 click 匹配状态
        click_match_states: List[str] = es_dict.get(EssentialStateKeyword.CLICK, None)
        if click_match_states:
            match_filter.extend([int(node_id_str) for node_id_str in click_match_states])

        # 【优化2】去重并设置uicomponent_set标志
        if match_filter:
            match_filter = list(set(match_filter))  # 去重
            uicomponent_set = 1
        tmp_assertion_states_annotated_total += len(match_filter) # total annotated assertion states, denominator(分母) for assertion-level recall


        if not click_match_states:
            # 遍历当前ui_state向后的所有连续的Oracle事件
            while node_id < len(exec_trace) and isinstance(exec_ui_state.action, OracleEvent):
                # 第一次比较时进行页面级别的检查
                if exec_ui_state.action.assert_accept is True:
                    
                    if first_cmp:
                        node_start_oracle_id = node_id

                        tmp_pages_annotated_hit = 1 # pages with annotated assertions and at least one hit, numerator(分子) for page-level recall and precision
                        page_hit = 1
                        # 检查activity匹配
                        if check_activity_match(gr_ui_state, exec_ui_state):
                            activity_hit = 1
                            
                        # 检查页面图片匹配
                        if check_page_image_match(gr_ui_state.screenshot_path, exec_ui_state.screenshot_path):
                            fuzzy_pageimg_hit = 1
                            
                        # 检查整个UI VH匹配
                        if compare_entire_ui_vh(gr_ui_state, exec_ui_state):
                            fuzzy_page_main_uicomps_hit = 1
                            
                        first_cmp = False
                        
                    else:
                        tmp_assertion_states_exec_total += 1
                        tmp_assertion_states_gen_total += 1

                    # 【优化3】只有在有匹配过滤器时才进行UI组件检查
                    if match_filter:
                        # 判断exec_ui_state.action是否在match_filter对应的节点字典中; if exec_ui_state.action.view is None (This is a problem exist in input_policy ), will return -1,-1,-1 (meaning no match at all)
                        nearFull_match, keyword_match, text_match = check_oracle_in_uicomponents(  
                            gr_ui_state, exec_ui_state.action, match_filter
                        )
                        
                        # 记录各种匹配结果
                        if nearFull_match != -1:
                            uicomponent_nearFull_hit = 1
                            # 【优化4】避免重复添加相同的匹配结果
                            if nearFull_match not in uicomp_nearFull_hit_list:
                                uicomp_nearFull_hit_list.append(nearFull_match)
                                
                        if keyword_match != -1:
                            uicomponent_keywords_hit = 1
                            if keyword_match not in uicomp_keywords_hit_list:
                                uicomp_keywords_hit_list.append(keyword_match)
                                tmp_assertion_states_annotated_hit += 1 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
                                
                        if text_match != -1:
                            uicomponent_text_hit = 1
                            if text_match not in uicomp_text_hit_list:
                                uicomp_text_hit_list.append(text_match)
                
                # 移动到下一个节点
                node_id += 1
                
                # 【优化5】边界检查并更新exec_ui_state
                if node_id < len(exec_trace):
                    exec_ui_state = exec_trace[node_id]
        
        # 如果有click匹配状态，需要向前检查，因为在点击事件中，页面状态的断言必须出现在点击事件之前的Oracle事件中
        if click_match_states:
            node_id_back = node_id - 1
            exec_ui_state_back = exec_trace[node_id_back]
            first_cmp = True  # 标记是否是第一次比较    
            # 遍历当前ui_state向后的所有连续的Oracle事件
            while node_id_back >= 0 and isinstance(exec_ui_state_back.action, OracleEvent):
                # 第一次比较时进行页面级别的检查
                if exec_ui_state_back.action.assert_accept is True:
                    if first_cmp:
                        node_start_oracle_id = node_id_back
                        
                        tmp_pages_annotated_hit = 1 # pages with annotated assertions and at least one hit, numerator(分子) for page-level recall and precision
                        page_hit = 1
                        # 检查activity匹配
                        if check_activity_match(gr_ui_state, exec_ui_state_back):
                            activity_hit = 1
                            
                        # 检查页面图片匹配
                        if check_page_image_match(gr_ui_state.screenshot_path, exec_ui_state_back.screenshot_path):
                            fuzzy_pageimg_hit = 1
                            
                        # 检查整个UI VH匹配
                        if compare_entire_ui_vh(gr_ui_state, exec_ui_state_back):
                            fuzzy_page_main_uicomps_hit = 1
                            
                        first_cmp = False
                    
                    # 【优化3】只有在有组件相关Assertion时才进行UI组件检查
                    if match_filter:
                        # 判断exec_ui_state.action是否在match_filter对应的节点字典中; if exec_ui_state.action.view is None (This is a problem exist in input_policy ), will return -1,-1,-1 (meaning no match at all)
                        nearFull_match, keyword_match, text_match = check_oracle_in_uicomponents(  
                            gr_ui_state, exec_ui_state_back.action, match_filter
                        )
                        
                        # 记录各种匹配结果
                        if nearFull_match != -1:
                            uicomponent_nearFull_hit = 1
                            # 【优化4】避免重复添加相同的匹配结果
                            if nearFull_match not in uicomp_nearFull_hit_list:
                                uicomp_nearFull_hit_list.append(nearFull_match)
                                
                        if keyword_match != -1:
                            uicomponent_keywords_hit = 1
                            if keyword_match not in uicomp_keywords_hit_list:
                                uicomp_keywords_hit_list.append(keyword_match)
                                tmp_assertion_states_annotated_hit += 1 # annotated assertion states with hit, numerator(分子) for assertion-level recall and precision
                                
                        if text_match != -1:
                            uicomponent_text_hit = 1
                            if text_match not in uicomp_text_hit_list:
                                uicomp_text_hit_list.append(text_match)
                
                # 移动到下一个节点
                node_id_back -= 1
                
                # 【优化5】边界检查并更新exec_ui_state
                if node_id_back >= 0:
                    exec_ui_state_back = exec_trace[node_id_back]
        
        # 创建Oracle命中结果元组
        oracle_hit_tuple = OracleHitTuple(
            episode=episode,
            node_start_id = node_start_id,
            node_start_oracle_id=node_start_oracle_id,
            oracle_dict=gr_ui_state.essential_state,
            page_hit=page_hit,
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
        
        # 【关键操作】将结果添加到hit_results列表中
        self.hit_results.append(oracle_hit_tuple)
        
        metrics = HitMetrics(
            pages_annotated_hit=tmp_pages_annotated_hit,
            assertion_states_exec_total=tmp_assertion_states_exec_total,
            assertion_states_gen_total=tmp_assertion_states_gen_total,
            assertion_states_annotated_total=tmp_assertion_states_annotated_total,
            assertion_states_annotated_hit=tmp_assertion_states_annotated_hit,
            is_assert_widget_set=uicomponent_set # 是否为组件相关的断言
        )
        # 如果当前执行状态的action不是OracleEvent，while循环未执行，node_id未变化
        # 则需要手动将node_id加1，确保继续处理下一个节点; 如果当前执行状态的action是OracleEvent，while循环执行后node_id指向有非OracleEvent的原匹配页面，需要指向下一个非OracleEvent的新页面，仍需再加1
        return node_id + 1, metrics



        


    def check_essential_state_match(  # In one gr_ui_state: UIState (.ess file), there are multiple essential keywords in some conditions
        self, gr_ui_state: UIState, exec_ui_state: UIState
    ) -> bool:
        assert (
            gr_ui_state.essential_state is not None
        ), f"essential state in {gr_ui_state} is None"

        es_dict = gr_ui_state.essential_state

        if self.fuzzy_match:  # fuzzy_match
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

        if self.exact_match:  # exact_match:
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
