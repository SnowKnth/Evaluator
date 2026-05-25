class CONFIG:
    # task metadata and the dataset
    # see docs at https://github.com/LlamaTouch/LlamaTouch/tree/main/dataset
    EPI_METADATA_PATH = "/data/wxd/LlamaTouch/dataset/llamatouch_task_metadata.tsv"
    GR_DATASET_PATH = "/data/wxd/LlamaTouch/dataset/llamatouch_dataset_0521"

    # agent exec trace path
    AUTOUI_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AutoUI_traces"
    AUTODROID_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AutoDroid_traces" #empty, only agg_plot.png exists
    APPAGENT_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/AppAgent_traces"
    COCOAGENT_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/CoCoAgent_traces"
    RASSDROID_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_0324"
    AUTODROID_DEEPSEEK_NO_SLEEP_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/AutoDroid/exec_output_llamatouch_autodroid_deepseek"
    exec_output_llamatouch_autodroid_deepseek_with_sleep_5s = "/data/wxd/LlamaTouch/AutoDroid/exec_output_llamatouch_autodroid_deepseek_with_sleep_5s"
    exec_output_llamatouch_autodroid_deepseek_scroll_text = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_autodroid_deepseek_scroll_text_04-05_1-495"

    RASSDROID_ORACLE_EXEC_TRACE_PATH = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_oracle"
    RASSDROID_ORACLE_EXEC_TRACE_PATH1 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_deepseek_oracle_03-30_1-250"
    RASSDROID_ORACLE_EXEC_TRACE_PATH2 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_autodroid_deepseek_scroll_text_04-16_234-495"
    
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_12 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-12-GPT4o-mini-ca"
    
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_05-11_1-"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11_few_assertion = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_few-assertion_deepseek_05-11_1-"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_05_11_simple_assertion = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_simple-assertion_deepseek_05-11_1-"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_06_13 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_06-13"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_06_30 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_06-30-no-adapt"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_05 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-05-no_subtask_version"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_09 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-09"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_19 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-19"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_23 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-23"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_07_29 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-29"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_09_20 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_09-20"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_10_19 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-19"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_10_22 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-22"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_10_25 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-25"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_10_27 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-27"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_10_29 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-29"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_01 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-01"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_03 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-03"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_04 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-04"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_06 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-06"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_08 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-08"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_10 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-10"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_12 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-12"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_13 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-13"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_14 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-14"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_25_01_15 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_25-01-15"

    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_15 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-15"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_17 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-17"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_19 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-19"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_22 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-22"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_26 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-26"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_29 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-29"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_01_31 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-31"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_26_02_02 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-02-02"

    # Agent-specific exec trace paths for evaluation
    # VASSODroid_Full_Follow_and_Adapt_InTime = "VASSODroid_26-02-02_Full_Follow_and_Adapt_InTime_Version"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FOLLOW_AND_ADAPT_INTIME = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-02-02"
    # VASSODroid_No_VASSO = "VASSODroid_26-01-26_No_VASSO_version"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_NO_VASSO = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-26"
    # VASSODroid_Full_First_Follow_then_Adapt = "VASSODroid_10-22_Full_First_Follow_then_Adapt_version"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FIRST_FOLLOW_THEN_ADAPT = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_10-22"
    # VASSODroid_No_Interaction_Validation = "VASSODroid_01-31_No_Interaction_Validation_version"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_NO_INTERACTION_VALIDATION = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-31"
    # AutoDroid = "AutoDroid_26-01-29"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_AUTODROID = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_26-01-29"
    
    # DeepSeek V3 versions
    # VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3 = "VASSODroid_Full_Follow_and_Adapt_InTime_DeepseekV3_07-09"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FOLLOW_AND_ADAPT_INTIME_DSV3 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_07-09"
    # VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3 = "VASSODroid_Full_First_Follow_then_Adapt_DeepseekV3_06-30"
    RASSDROID_ORACLE_EXEC_TRACE_PATH_FULL_FIRST_FOLLOW_THEN_ADAPT_DSV3 = "/data/wxd/LlamaTouch/RASSDroid/exec_output_llamatouch_RASSDroid_deepseek_06-30-no-adapt"

    # human eval result path
    AUTOUI_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_autoui.csv"
    AUTODROID_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_autodroid.csv"
    APPAGENT_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_appagent.csv"
    COCOAGENT_HUMANEVAL_PATH = "/data/wxd/LlamaTouch/agent_exec_traces/human_cocoagent.csv"
