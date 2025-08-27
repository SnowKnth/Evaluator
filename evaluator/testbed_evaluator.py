import logging
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
                    # click_match_states: List[str] = ui_state.get(EssentialStateKeyword.CLICK, None)
                    i = self.calc_hit_results(episode, ui_state, exec_trace, i) # # 114 (without), 100 with this line ; 排查 Oracle.view is None 的原因
                    break

            if gr_ui_state_matched:
                continue
            else:
                return False, None

        return True, None

    def post_evaluation_hook(self) -> None:
        """
        Dump hit results after evaluation.
        """
        print(f"DEBUG: hit_results length: {len(self.hit_results)}")  # 添加调试信息
        if self.hit_results:
            print(f"DEBUG: First hit result: {self.hit_results[0]}")
        
        file_name = f"dumped_stats/oracle_hit_results_{self.evaluator_name}_{self.agent.agent_name}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.csv"
        self.dump_hit_results(file_name)

    def dump_hit_results(self, filename: str = "oracle_hit_results.csv"):
        import csv
        print(f"DEBUG: Trying to dump {len(self.hit_results)} results to {filename}")  # 添加调试信息
        if not self.hit_results:
            print("DEBUG: hit_results is empty, not creating file")  # 添加调试信息
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
        activity_hit: int = 0  # activity匹配命中
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

        # 获取ground-truth的essential state字典
        es_dict = gr_ui_state.essential_state
        first_cmp = True  # 标记是否是第一次比较
        node_start_id = node_id #记录gr_ui_state匹配开始的节点ID
        node_start_oracle_id = -1  # 记录Oracle事件开始的节点ID
        
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

        # 遍历当前ui_state向后的所有连续的Oracle事件
        while node_id < len(exec_trace) and isinstance(exec_ui_state.action, OracleEvent):
            # 第一次比较时进行页面级别的检查
            if exec_ui_state.action.assert_accept is True:
                if first_cmp:
                    node_start_oracle_id = node_id
                    
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
                            
                    if text_match != -1:
                        uicomponent_text_hit = 1
                        if text_match not in uicomp_text_hit_list:
                            uicomp_text_hit_list.append(text_match)
            
            # 移动到下一个节点
            node_id += 1
            
            # 【优化5】边界检查并更新exec_ui_state
            if node_id < len(exec_trace):
                exec_ui_state = exec_trace[node_id]
        
        # 如果有click匹配状态，继续向前检查
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
        # 如果当前执行状态的action不是OracleEvent，while循环未执行，node_id未变化
        # 则需要手动将node_id加1，确保继续处理下一个节点
        if not isinstance(exec_ui_state.action, OracleEvent):
            return node_id + 1
        
        return node_id

        


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
