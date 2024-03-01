import os
import re
from enum import Enum
from typing import Any, Dict, List, NamedTuple, Tuple

from action_type import ActionType


class Agent(Enum):
    APPAGENT = "AppAgent"
    AUTOUI = "Auto-UI"
    AUTODROID = "AutoDroid"
    COGAGENT = "CogAgent"


class TaskCategory(Enum):
    GENERAL = "general"
    GOOGLEAPPS = "googleapps"
    INSTALL = "install"
    WEBSHOPPING = "webshopping"
    GENERATED = "generated"


GROUNDTRUTH_DATASET_PATH = "/data/yyh/mobile/capture/AITW_decode"


ACTION_SPACE = {
    "Home键": ActionType.PRESS_HOME,
    "Back键": ActionType.PRESS_BACK,
    "点击事件": ActionType.DUAL_POINT,
    "滑动事件": ActionType.DUAL_POINT,
    "键盘输入": ActionType.TYPE,
}


class UIState(NamedTuple):
    screenshot: Any
    vh: str
    action: Dict


TaskTrace = List[UIState]

# =========================================================================
# deprecated
# =========================================================================
AGENT_EXEC_TRACE_PATH = {
    Agent.APPAGENT: "/data/wangshihe/AgentTestbed/AppAgent-AITW",
    Agent.AUTOUI: "/data/zzh/mobile-agent/Auto-UI/agentenv/agent_result",
    Agent.AUTODROID: None,
    Agent.COGAGENT: None,
}

APPAGENT_CATEGORY_TO_TRACE_FOLDER_NAME = {
    TaskCategory.GENERAL: "tasks-240215-1-general",
    TaskCategory.GOOGLEAPPS: "tasks-240216-1-googleapp",
    TaskCategory.INSTALL: "tasks-240215-3-install",
    TaskCategory.WEBSHOPPING: "tasks-240215-2-webshopping",
    TaskCategory.GENERATED: None,
}


def get_agent_exec_trace_folder(agent_name, episode) -> str:
    """Get the folder of agent execution trace for one specific episode"""
    category = DatasetHelper().get_category_by_episode(episode)
    if agent_name == Agent.APPAGENT:
        trace_folder = os.path.join(
            AGENT_EXEC_TRACE_PATH[Agent.APPAGENT],
            APPAGENT_CATEGORY_TO_TRACE_FOLDER_NAME[category],
            episode,
            episode,
            "captured_data",
        )
    elif agent_name == Agent.AUTOUI:
        trace_folder = os.path.join(
            AGENT_EXEC_TRACE_PATH[Agent.AUTOUI],
            category,
            episode,
            "captured_data",
        )
    else:
        pass

    """
    The trace path contains the following folders:
        - screenshot
        - view_hierarchy
        - activity
        - action
        - installed_apps
        - xml
    """
    return trace_folder


# =========================================================================


def load_groundtruth_trace(
    category: TaskCategory,
) -> Dict[str, List[UIState]]:
    """
    Return: {
        "episode_id_1": [(screenshot_1_1, XML_1_1, action_1_1), (screenshot_1_2, XML_1_2, action_1_2), ...],
        "episode_id_2": [(screenshot_2_1, XML_2_1, action_2_1), (screenshot_2_2, XML_2_2, action_2_2), ...],
        ...
    }
    """
    groundtruth_trace_folder = os.path.join(GROUNDTRUTH_DATASET_PATH, category)
    gt_trace_dict = {}
    dirs = [
        d
        for d in os.listdir(groundtruth_trace_folder)
        if os.path.isdir(os.path.join(groundtruth_trace_folder, d))
    ]
    dirs.sort()
    for dir in dirs:
        path = os.path.join(groundtruth_trace_folder, dir)
        ep_id_path = os.path.join(path, "instruction.txt")
        with open(ep_id_path, "r") as f:
            ep_id = f.readline().strip()

        ep_trace_list = get_trace_by_path(path)
        gt_trace_dict[ep_id] = ep_trace_list

    return gt_trace_dict


def get_trace_by_path(path: str) -> List[UIState]:
    ep_trace_list: TaskTrace = []
    files = [
        f for f in os.listdir(path) if f.endswith(".png") and f.find("png_image") != -1
    ]
    files.sort()

    action_path = os.path.join(path, "eventStructs.txt")
    action_list = []
    with open(action_path, "r") as f:
        action_texts = f.readlines()

    for action_text in action_texts:
        action_type = re.search("【(?P<action_type>.+)】", action_text).groupdict()[
            "action_type"
        ]
        if action_type == "Home键" or action_type == "Back键":
            action_list.append({"action_type": ACTION_SPACE[action_type]})
        elif action_type == "点击事件":
            pattern = re.compile(
                "屏幕大小：（w(?P<screen_wight>\d+)，h(?P<screen_height>\d+)），触摸位置：（x(?P<position_1_x>\d+)，y(?P<position_1_y>\d+)）"
            )
            re_dict = re.search(pattern, action_text).groupdict()
            action_list.append(
                {
                    "action_type": ACTION_SPACE[action_type],
                    "begin_x": re_dict["position_1_x"],
                    "begin_y": re_dict["position_1_y"],
                    "end_x": re_dict["position_1_x"],
                    "end_y": re_dict["position_1_y"],
                }
            )
        elif action_type == "滑动事件":
            pattern = re.compile(
                "屏幕大小：（w(?P<screen_wight>\d+)，h(?P<screen_height>\d+)），起始位置：（x(?P<position_1_x>\d+)，y(?P<position_1_y>\d+)），结束位置：（x(?P<position_2_x>\d+)，y(?P<position_2_y>\d+)）"
            )
            re_dict = re.search(pattern, action_text).groupdict()
            action_list.append(
                {
                    "action_type": ACTION_SPACE[action_type],
                    "begin_x": re_dict["position_1_x"],
                    "begin_y": re_dict["position_1_y"],
                    "end_x": re_dict["position_2_x"],
                    "end_y": re_dict["position_2_y"],
                }
            )
        elif action_type == "键盘输入":
            pattern = re.compile("【键盘输入】(?P<text>.+)")
            text = re.search(pattern, action_text).groupdict()["text"]
            action_list.append({"action_type": ACTION_SPACE[action_type], "text": text})
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    for file, action in zip(files, action_list):
        img_path = os.path.join(path, file)
        xml_path = os.path.join(path, file.replace("png_image.png", "png_xml.txt"))
        with open(xml_path, "r") as f:
            xml_text = f.read()
        ep_trace_list.append(UIState(screenshot=img_path, vh=xml_text, action=action))

    return ep_trace_list


def load_groundtruth_trace_by_episode(episode: str):
    category: TaskCategory = DatasetHelper().get_category_by_episode(episode)
    print(f"episode: {episode}, category: {category}")
    return load_groundtruth_trace(category)[episode]


def load_trace(trace_folder) -> List[Tuple[Any, str, Dict]]:
    """
    TODO
    Return: [[screenshot1, XML1, action1], [screenshot2, XML2, action2], ...]
    """
    return [(None, None, None), (None, None, None)]


class DatasetHelper:
    """A singleton class to help load task metadata from the dataset."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatasetHelper, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        # load task metadata
        self.epi_to_category_file = "data/epi_to_category.csv"
        assert os.path.exists(
            self.epi_to_category_file
        ), f"The file {self.epi_to_category_file} does not exist"
        self.epi_metadata_dict = {}
        self.init_epi_to_category()

    def init_epi_to_category(self):
        """
        Load episode metadata from the file {self.epi_to_category_file}
        Format: {
            "episode": {
                "category": xx,
                "task_description": xx,
            },
            ...
        }
        """
        with open(self.epi_to_category_file, "r") as f:
            next(f)  # f is an iterable file object; skip the header
            for line in f:
                epi, category, task_description = line.strip().split(",", maxsplit=2)
                self.epi_metadata_dict[epi] = {
                    "category": category,
                    "task_description": task_description,
                }

    def get_all_episodes(self) -> List[str]:
        return self.epi_metadata_dict.keys()

    def get_task_decsription_by_episode(self, episode) -> str:
        return self.epi_metadata_dict[episode]["task_description"]

    def get_category_by_episode(self, episode) -> TaskCategory:
        return self.epi_metadata_dict[episode]["category"]


if __name__ == "__main__":
    gt_trace = load_groundtruth_trace(TaskCategory.GENERAL)
    # print(gt_trace)
