import logging
import json
import re
from typing import Dict, List, Optional

import imagehash
from lxml import etree
from PIL import Image

from ..common.action_type import ActionType
from ..task_trace import EssentialStateKeyword, UIState
from droidbot.input_event import OracleEvent




def _get_image_patch(image: Image, bounds: List[int]) -> Image:
    left, top, right, bottom = bounds
    return image.crop((left, top, right, bottom))


def _check_img_exact_match(
    annotated_ui_node: Dict,
    gr_screenshot_path: str,
    exec_screenshot_path: str,
    image_similarity_bound: Optional[int] = 5,
) -> bool:
    """
    Compare whether the image patch of the annotated UI component matches
    the corresponding image patch in the execution trace.

    Args:
        annotated_ui_node: annotated essential state that represents a UI node
        gr_screenshot_path: screenshot path of the annotated UI
        exec_screenshot_path: screenshot path of the execution UI
        image_similarity_bound: threshold to determine whether the image patches are similar

    Return:
        boolean value indicating whether the image patch of the annotated UI component
        and the execution UI component matches
    """
    gr_bounds = annotated_ui_node["bounds"]
    assert gr_bounds is not None

    gr_screen_width, gr_screen_height = 0, 0
    exec_screen_width, exec_screen_height = 0, 0
    with Image.open(gr_screenshot_path) as img:
        gr_screen_width, gr_screen_height = img.size
    with Image.open(exec_screenshot_path) as img:
        exec_screen_width, exec_screen_height = img.size

    gr_l, gr_t, gr_r, gr_b = map(
        int, re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", gr_bounds)[0]
    )
    exec_l, exec_t, exec_r, exec_b = (
        gr_l * exec_screen_width / gr_screen_width,
        gr_t * exec_screen_height / gr_screen_height,
        gr_r * exec_screen_width / gr_screen_width,
        gr_b * exec_screen_height / gr_screen_height,
    )

    annotate_image_patch: Image = _get_image_patch(
        Image.open(gr_screenshot_path), [gr_l, gr_t, gr_r, gr_b]
    )
    exec_image_patch: Image = _get_image_patch(
        Image.open(exec_screenshot_path), [exec_l, exec_t, exec_r, exec_b]
    )

    gr_hash = imagehash.phash(annotate_image_patch)
    exec_hash = imagehash.phash(exec_image_patch)

    if gr_hash - exec_hash > image_similarity_bound:
        logging.info(
            f"[image] match fail: hamming distance: {gr_hash-exec_hash}, '{gr_screenshot_path}' with '{exec_screenshot_path}'"
        )
        return False

    logging.info(
        f"[image] match success: hamming distance: {gr_hash-exec_hash}, '{gr_screenshot_path}' with '{exec_screenshot_path}'"
    )
    return True

def check_page_image_match( gr_screenshot_path: str, exec_screenshot_path: str, image_similarity_bound: Optional[int] = 5,) -> bool:
    with Image.open(gr_screenshot_path) as img_gr, Image.open(exec_screenshot_path) as img_exec:
        gr_hash = imagehash.phash(img_gr)
        exec_hash = imagehash.phash(img_exec)

        if gr_hash - exec_hash > image_similarity_bound:
            logging.info(
                f"[image] match fail: hamming distance: {gr_hash-exec_hash}, '{gr_screenshot_path}' with '{exec_screenshot_path}'"
            )
            return False

        logging.info(
            f"[image] match success: hamming distance: {gr_hash-exec_hash}, '{gr_screenshot_path}' with '{exec_screenshot_path}'"
        )
        return True


def _check_exact_single_node_match(
    annotated_ui_node: Dict, exec_ui_node: etree.ElementTree
) -> bool:
    """Compare whether current UI representation in the execution trace matches
    the annotated UI component. Find one node in the execution UI tree (traverse) that matches
    the annotated one. The annotated UI component should be matched only when all attributes like
    'text', 'content-desc', 'checked', 'resource-id', etc. are matched.

    Args:
        annotated_ui_node: annotated essential state that represents a UI node
        exec_ui_node: UI hierarchy tree of one screen in the execution trace

    Return:
        boolean value indicating whether there is one node in the current UI
        matches the annotated one
    """
    checked_attrs = [
        "class",
        "text",
        "resource-id",
        "content-desc",
        "enabled",
        "checked",
        "checkable",
        "selected",
        "focused",
        "focusable",
        "clickable",
        "long-clickable",
        "password",
        "scrollable",
    ]

    # for each node, we need to check all attributes listed above
    for node in exec_ui_node.iter():

        # modify this variable to False until current node doesnot match the annotated one
        find_node_match = True
        for attr in checked_attrs:
            assert attr in annotated_ui_node
            node_attr = node.get(attr)
            annotate_ui_node_attr = annotated_ui_node.get(attr)

            if node_attr is None:
                if annotate_ui_node_attr == None or annotate_ui_node_attr in [
                    "",
                    " ",
                    "null",
                ]:
                    continue
                else:
                    find_node_match = False
                    break

            if node_attr == "true":
                node_attr = True
            if node_attr == "false":
                node_attr = False
            if attr == "text" or attr == "content-desc":
                annotate_ui_node_attr = annotate_ui_node_attr.lower()
                node_attr = node_attr.lower()

            if node_attr == annotate_ui_node_attr:
                continue
            else:
                find_node_match = False
                break

        if find_node_match:
            return True
    return False


def check_uicomponent_match(gr_ui_state: UIState, exec_ui_state: UIState) -> bool:
    """Exact match on two UI components"""
    match_node_ids: List[str] = gr_ui_state.essential_state[EssentialStateKeyword.EXACT]
    null_state = ["", " ", "null", None]

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    exec_ui_tree = etree.parse(exec_ui_state.vh_path, parser)
    # by wxd, .vh and .json are different, .json has less dict object in the array than in .vh 
    for node_id in match_node_ids:
        node_id = int(node_id)
        gr_vh_simp_ui_json_path = gr_ui_state.vh_simp_ui_json_path
        annotated_ui_repr: Dict = json.load(
            open(gr_vh_simp_ui_json_path, "r", encoding="utf-8")
        )[node_id]

        if (
            annotated_ui_repr.get("text", None) in null_state
            and annotated_ui_repr.get("content-desc", None) in null_state
        ):
            # cmp img batch
            if not _check_img_exact_match(
                annotated_ui_repr,
                gr_ui_state.screenshot_path,
                exec_ui_state.screenshot_path,
            ):
                return False

        else:
            # if there is one annotated UI component (indicated by node_id) has no
            # matched counterpart, directly return False to indicate
            if not _check_exact_single_node_match(annotated_ui_repr, exec_ui_tree):
                # logging.info(f"[textbox] match failed: '{gr_ui_state.screenshot_path}', essential state: {annotated_ui_repr}")
                return False

    logging.info(
        f"[textbox] match success: '{gr_ui_state.screenshot_path}' with '{exec_ui_state.screenshot_path}'"
    )
    return True

def extract_checked_attrs_from_event_view(view: dict) -> dict:
    """
    从event['view']字典中提取checked_attrs对应的字段并做名称映射，返回新dict。
    """
    checked_attrs = [
        "class",
        "text",
        "resource-id",
        "content-desc",
        "enabled",
        "checked",
        "checkable",
        "selected",
        "focused",
        "focusable",
        "clickable",
        "long-clickable",
        "password",
        "scrollable",
    ]
    # 映射关系：checked_attrs字段 -> event['view']中的字段
    key_map = {
        "resource-id": "resource_id",
        "content-desc": "content_description",
        "long-clickable": "long_clickable",
        "password": "is_password",
    }
    result = {}
    for attr in checked_attrs:
        src_key = key_map.get(attr, attr)
        result[attr] = view.get(src_key)
    return result

from typing import Tuple
# judge whether exec_ui_state.action is in dict related to node in match_filter
def check_oracle_in_uicomponents(gr_ui_state: UIState, oracle: OracleEvent, match_filter: List[int]) -> Tuple[int, int, int]:
    
    nearFull_match = -1
    keyword_match = -1
    text_match = -1
    if oracle.view is None:
        return nearFull_match, keyword_match, text_match
    trans_oracle_view = extract_checked_attrs_from_event_view(oracle.view)
    checked_attrs_nearfull = [
        "class",
        "text",
        "resource-id",
        "content-desc",
        "enabled",
        "checked",
        "checkable",
        "selected",
        "focused",
        "focusable",
        "clickable",
        "long-clickable",
        "password",
        "scrollable",
    ]
    checked_attrs_keywords = ["text","resource-id","content-desc"]
    checked_attrs_text = ["text"]
    
    # Load UI nodes once to avoid repeated I/O in the loop
    gr_vh_simp_ui_json_path = gr_ui_state.vh_simp_ui_json_path
    with open(gr_vh_simp_ui_json_path, "r", encoding="utf-8") as f:
        all_ui_nodes = json.load(f)
    # 根据checked_attrs_nearfull、checked_attrs_keywords、checked_attrs_text的字段进行匹配，判断trans_oracle_view是否在gr_ui_state的match_filter对应的节点中，并记录匹配结果（match_filter中的节点id）到nearFull_match、keyword_match、text_match
    for node_id_str in match_filter:
        node_id = int(node_id_str)
        annotated_ui_repr: Dict = all_ui_nodes[node_id]

        # Check for matches only if they haven't been found yet.
        # This prevents overwriting with a later, potentially less relevant match.
        if nearFull_match == -1:
            if all(_attrs_equal(annotated_ui_repr.get(attr), trans_oracle_view.get(attr), attr) for attr in checked_attrs_nearfull):
                nearFull_match = node_id

        if keyword_match == -1:
            if all(_attrs_equal(annotated_ui_repr.get(attr), trans_oracle_view.get(attr), attr) for attr in checked_attrs_keywords):
                keyword_match = node_id

        if text_match == -1:
            if all(_attrs_equal(annotated_ui_repr.get(attr), trans_oracle_view.get(attr), attr) for attr in checked_attrs_text):
                text_match = node_id
        
        # Optimization: if all levels of matches are found, we can exit early.
        if nearFull_match != -1 and keyword_match != -1 and text_match != -1:
            break

    return nearFull_match, keyword_match, text_match
        

def check_activity_match(gr_ui_state: UIState, exec_ui_state: UIState) -> bool:
    if gr_ui_state.activity == "null":
        return True
        # raise Exception("[activity] match required; but annotated activity is null.")

    match = True if exec_ui_state.activity in gr_ui_state.activity else False
    if match:
        logging.info(
            f"[actvity] match success: '{gr_ui_state.activity}' with '{exec_ui_state.activity}'"
        )
    return match


def check_type_match(gr_ui_state: UIState, exec_ui_state: UIState) -> bool:
    if exec_ui_state.action.action_type != ActionType.TYPE:
        return False

    if (
        exec_ui_state.action.typed_text
        == gr_ui_state.essential_state[EssentialStateKeyword.TYPE][0]
    ):
        return True
    else:
        return False


def check_click_match(gr_ui_state: UIState, exec_ui_state: UIState) -> bool:
    """
    based on the clicked item xpath in gr_ui_state.essential_state, find
    the corresponding node in the exec_ui_state and check if the click point in the node.
    """
    if not hasattr(exec_ui_state.action, 'action_type'):
        return False
    if exec_ui_state.action.action_type != ActionType.DUAL_POINT:
        return False

    if exec_ui_state.action.touch_point_yx != exec_ui_state.action.lift_point_yx:
        return False

    gr_click_id = int(gr_ui_state.essential_state[EssentialStateKeyword.CLICK][0])
    gr_vh_simp_ui_json_path = gr_ui_state.vh_simp_ui_json_path
    gr_click_xpath: str = json.load(
        open(gr_vh_simp_ui_json_path, "r", encoding="utf-8")
    )[gr_click_id]["xpath"] # by wxd, no xpath string exist?

    parser = etree.XMLParser(recover=True, encoding="utf-8")
    exec_ui_tree = etree.parse(exec_ui_state.vh_path, parser)

    found_nodes = exec_ui_tree.xpath(gr_click_xpath)
    if len(found_nodes) == 0:
        raise AssertionError("No corresponding nodes found in the execution UI state.")

    bounds = found_nodes[0].get("bounds")
    if not bounds:
        raise AssertionError("No bounds found for the corresponding node.")
    left, top, right, bottom = map(
        int, re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)[0]
    )

    screen_width, screen_height = 0, 0
    with Image.open(exec_ui_state.screenshot_path) as img:
        screen_width, screen_height = img.size

    # screen_width, screen_height = Image.open(exec_ui_state.screenshot_path).size
    y = exec_ui_state.action.touch_point_yx[0] * screen_height
    x = exec_ui_state.action.touch_point_yx[1] * screen_width

    if left <= x <= right and top <= y <= bottom:
        logging.info(
            f"[click] match success: click action:{x,y}, '{gr_ui_state.vh_path}' with '{exec_ui_state.vh_path}'"
        )
        return True
    else:
        logging.info(
            f"[click] match failed: click action:{x,y}, '{gr_ui_state.vh_path}' with '{exec_ui_state.vh_path}'"
        )
        return False

def _normalize_attr_value(value):
    """标准化属性值，处理None和空字符串的差异"""
    if value is None or value == '' or value == 'null':
        return None
    if isinstance(value, str):
        return value.strip()
    return value

def _attrs_equal(val1, val2, attr_name: str = None) -> bool:
    """比较两个属性值是否相等，处理特殊情况"""
    val1_norm = _normalize_attr_value(val1)
    val2_norm = _normalize_attr_value(val2)
    
    # 对于text和content-desc属性，进行大小写不敏感比较
    if attr_name in ["text", "content-desc"] and val1_norm and val2_norm:
        return str(val1_norm).lower() == str(val2_norm).lower()
    
    return val1_norm == val2_norm

