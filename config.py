class CONFIG:
    # task metadata and the dataset
    # see docs at https://github.com/LlamaTouch/LlamaTouch/tree/main/dataset
    EPI_METADATA_PATH = "/data/wxd/LlamaTouch/dataset/llamatouch_task_metadata.tsv"
    GR_DATASET_PATH = "/data/wxd/LlamaTouch/dataset/llamatouch_dataset_0521"

    # agent exec trace path
    AUTOUI_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AutoUI_traces"
    AUTODROID_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AutoDroid_traces"
    APPAGENT_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AppAgent_traces"
    COCOAGENT_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/CoCoAgent_traces"
    RASSDROID_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_0324"
    AUTODROID_DEEPSEEK_NO_SLEEP_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/AutoDroid/exec_output_llamatouch_autodroid_deepseek"
    exec_output_llamatouch_autodroid_deepseek_with_sleep_5s = "/data/wxd/LlamaTouch/AutoDroid/exec_output_llamatouch_autodroid_deepseek_with_sleep_5s"
    exec_output_llamatouch_autodroid_deepseek_scroll_text = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_autodroid_deepseek_scroll_text_04-05_1-495"

    RASSDROID_ORACLE_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_oracle"
    RASSDROID_ORACLE_EXEC_TRACE_PATH1 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_oracle_03-30_1-250"
    RASSDROID_ORACLE_EXEC_TRACE_PATH2 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_autodroid_deepseek_scroll_text_04-16_234-495"
    
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_05-11_1-"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11_few_assertion = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_few-assertion_deepseek_05-11_1-"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11_simple_assertion = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_simple-assertion_deepseek_05-11_1-"
    # human eval result path
    AUTOUI_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_autoui.csv"
    AUTODROID_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_autodroid.csv"
    APPAGENT_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_appagent.csv"
    COCOAGENT_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_cocoagent.csv"
