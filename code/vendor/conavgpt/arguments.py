import argparse
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(
        description='Multi-Agent-Semantic-Exploration')

    # General Arguments
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--auto_gpu_config', type=int, default=0)
    parser.add_argument('--total_num_scenes', type=str, default="auto")
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--sim_gpu_id", type=int, default=0,
                        help="gpu id on which scenes are loaded")
    parser.add_argument("--sem_gpu_id", type=int, default=0,
                        help="""gpu id for semantic model,""")
    parser.add_argument("--llm_gpu_id", type=int, default=-1,
                        help="gpu id for local LLM; defaults to --sem_gpu_id when negative")

    # Logging, loading models, visualization
    parser.add_argument('--log_interval', type=int, default=10,
                        help="""log interval, one log per n updates
                                (default: 10) """)
    parser.add_argument('--save_interval', type=int, default=1,
                        help="""save interval""")
    parser.add_argument('-d', '--dump_location', type=str, default="./tmp/",
                        help='path to dump models and log (default: ./tmp/)')
    parser.add_argument('--exp_name', type=str, default="exp1",
                        help='experiment name (default: exp1)')
    parser.add_argument('--save_periodic', type=int, default=500000,
                        help='Model save frequency in number of updates')
    parser.add_argument('-v', '--visualize', type=int, default=0,
                        help="""1: Render the observation and
                                   the predicted semantic map,
                                2: Render the observation with semantic
                                   predictions and the predicted semantic map
                                (default: 0)""")
    parser.add_argument('--print_images', type=int, default=0,
                        help='1: save visualization as images')

    # Environment, dataset and episode specifications
    parser.add_argument('-efw', '--env_frame_width', type=int, default=640,
                        help='Frame width (default:640)')
    parser.add_argument('-efh', '--env_frame_height', type=int, default=480,
                        help='Frame height (default:480)')
    parser.add_argument('-fw', '--frame_width', type=int, default=160,
                        help='Frame width (default:160)')
    parser.add_argument('-fh', '--frame_height', type=int, default=120,
                        help='Frame height (default:120)')
    parser.add_argument('-el', '--max_episode_length', type=int, default=500,
                        help="""Maximum episode length""")
    parser.add_argument('--max_episodes', type=int,
                        default=int(os.environ.get("MAX_EPISODES", "0")),
                        help='maximum number of episodes to run; 0 means all episodes in split')
    parser.add_argument('--start_episode_index', type=int,
                        default=int(os.environ.get("START_EPISODE_INDEX", "0")),
                        help='number of Habitat iterator episodes to skip before evaluation')
    parser.add_argument("--task_config", type=str,
                        default="tasks/multi_objectnav_hm3d.yaml",
                        help="path to config yaml containing task information")
    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")
    parser.add_argument('--camera_height', type=float, default=0.88,
                        help="agent camera height in metres")
    parser.add_argument('--hfov', type=float, default=79.0,
                        help="horizontal field of view in degrees")
    parser.add_argument('--turn_angle', type=float, default=30,
                        help="Agent turn angle in degrees")
    parser.add_argument('--min_depth', type=float, default=0.5,
                        help="Minimum depth for depth sensor in meters")
    parser.add_argument('--max_depth', type=float, default=5.0,
                        help="Maximum depth for depth sensor in meters")
    parser.add_argument('--success_dist', type=float, default=1.0,
                        help="success distance threshold in meters")
    parser.add_argument('--floor_thr', type=int, default=50,
                        help="floor threshold in cm")
    parser.add_argument('--min_d', type=float, default=1.5,
                        help="min distance to goal during training in meters")
    parser.add_argument('--max_d', type=float, default=100.0,
                        help="max distance to goal during training in meters")

    # Model Hyperparameters
    parser.add_argument('--agent', type=str, default="sem_exp")
    parser.add_argument('--num_global_steps', type=int, default=20,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--num_local_steps', type=int, default=25,
                        help="""Number of steps the local policy
                                between each global step""")
    parser.add_argument('--num_sem_categories', type=int, default=16)
    parser.add_argument('--sem_pred_prob_thr', type=float, default=0.9,
                        help="Semantic prediction confidence threshold")

    # Mapping
    parser.add_argument('--global_downscaling', type=int, default=1)
    parser.add_argument('--vision_range', type=int, default=100)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--du_scale', type=int, default=1)
    parser.add_argument('--map_size_cm', type=int, default=2400)
    parser.add_argument('--cat_pred_threshold', type=float, default=5.0)
    parser.add_argument('--map_pred_threshold', type=float, default=1.0)
    parser.add_argument('--exp_pred_threshold', type=float, default=1.0)
    parser.add_argument('--collision_threshold', type=float, default=0.10)

    # train_se_frontier
    parser.add_argument('--use_gtsem', type=int, default=0)
    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--gpt_type', type=int, default=1,
                        help="""0: text-davinci-003
                                1: gpt-3.5-turbo
                                2: gpt-4
                                (default: 1)""")
    parser.add_argument('--brain', type=str, default="helicase",
                        help='navigation brain name kept for script compatibility')
    parser.add_argument('--llm_path', type=str, default=None,
                        help='local text LLM path; overrides --gpt_type model defaults')
    parser.add_argument('--brain_backend', type=str,
                        choices=("local", "siliconflow", "deepseek"),
                        default=os.environ.get("BRAIN_BACKEND", "local"),
                        help='MindNav decision backend')
    parser.add_argument('--brain_model', type=str,
                        default=os.environ.get("BRAIN_MODEL", os.environ.get("DEEPSEEK_MODEL", "Pro/MiniMaxAI/MiniMax-M2.5")),
                        help='remote brain model name for API backends')
    parser.add_argument('--brain_base_url', type=str,
                        default=os.environ.get("BRAIN_BASE_URL", os.environ.get("DEEPSEEK_BASE_URL", None)),
                        help='OpenAI-compatible base URL for API brain backends')
    parser.add_argument('--deepseek_thinking', type=str,
                        choices=("disabled", "enabled"),
                        default=os.environ.get("DEEPSEEK_THINKING", "disabled"),
                        help='DeepSeek reasoning mode; keep disabled for non-thinking responses')
    parser.add_argument('--mindnav_mode', type=str,
                        choices=("kg", "semantic", "heuristic"),
                        default=os.environ.get("MINDNAV_MODE", "kg"),
                        help='MindNav frontier assignment mode')
    parser.add_argument('--mindnav_config', type=str,
                        default=os.environ.get("MINDNAV_CONFIG", None),
                        help='optional MindNav YAML config with KG/tool-calling parameters')
    parser.add_argument('--brain_max_tokens', type=int,
                        default=int(os.environ.get("BRAIN_MAX_TOKENS", "512")),
                        help='maximum new tokens for MindNav brain responses')
    parser.add_argument('--mindnav_target_tau', type=float,
                        default=float(os.environ.get("MINDNAV_TARGET_TAU", "0.5")),
                        help='certainty threshold for direct target pursuit')
    parser.add_argument('--jsonl_log', type=str, default=None,
                        help='optional per-episode JSONL metrics path')
    parser.add_argument('--append_jsonl', type=int,
                        default=int(os.environ.get("APPEND_JSONL", "0")),
                        help='1: append to existing JSONL metrics instead of truncating')
    parser.add_argument('--method_name', type=str, default=None,
                        help='method name written to JSONL metrics')
    parser.add_argument('--kg_trace_dir', type=str,
                        default=os.environ.get("KG_TRACE_DIR", None),
                        help='optional directory for MindNav KG JSONL traces and plots')
    parser.add_argument('--kg_trace_plots', type=int,
                        default=int(os.environ.get("KG_TRACE_PLOTS", "1")),
                        help='1: render KG topology and map-overlay plots when tracing')

    parser.add_argument('--semantic_boost_backend', type=str,
                        choices=("none", "grounded_sam"),
                        default=os.environ.get(
                            "SEMANTIC_BOOST_BACKEND",
                            os.environ.get("SEMANTIC_BOOST", "none"),
                        ),
                        help='optional semantic-map enhancer')
    parser.add_argument('--semantic_boost_model_id', type=str,
                        default=os.environ.get(
                            "SEMANTIC_BOOST_MODEL_ID",
                            "IDEA-Research/grounding-dino-base",
                        ),
                        help='HuggingFace zero-shot detector model id')
    parser.add_argument('--semantic_boost_sam_type', type=str,
                        default=os.environ.get("SEMANTIC_BOOST_SAM_TYPE", "vit_b"),
                        help='SAM model type from segment-anything')
    parser.add_argument('--semantic_boost_sam_checkpoint', type=str,
                        default=os.environ.get(
                            "SEMANTIC_BOOST_SAM_CHECKPOINT",
                            "/home/huaziheng/models/vision/sam/sam_vit_b_01ec64.pth",
                        ),
                        help='SAM checkpoint path')
    parser.add_argument('--semantic_boost_device', type=str,
                        default=os.environ.get("SEMANTIC_BOOST_DEVICE", None),
                        help='device for semantic boost models; defaults to semantic model device')
    parser.add_argument('--semantic_boost_interval', type=int,
                        default=int(os.environ.get("SEMANTIC_BOOST_INTERVAL", "5")),
                        help='run semantic boost once every N local frames')
    parser.add_argument('--semantic_boost_box_threshold', type=float,
                        default=float(os.environ.get("SEMANTIC_BOOST_BOX_THRESHOLD", "0.25")),
                        help='GroundingDINO box threshold')
    parser.add_argument('--semantic_boost_text_threshold', type=float,
                        default=float(os.environ.get("SEMANTIC_BOOST_TEXT_THRESHOLD", "0.20")),
                        help='GroundingDINO text threshold')
    parser.add_argument('--semantic_boost_sam_iou_threshold', type=float,
                        default=float(os.environ.get("SEMANTIC_BOOST_SAM_IOU_THRESHOLD", "0.75")),
                        help='minimum SAM mask quality score')
    parser.add_argument('--semantic_boost_min_mask_area', type=int,
                        default=int(os.environ.get("SEMANTIC_BOOST_MIN_MASK_AREA", "20")),
                        help='discard masks smaller than this many pixels')
    parser.add_argument('--semantic_boost_max_mask_frac', type=float,
                        default=float(os.environ.get("SEMANTIC_BOOST_MAX_MASK_FRAC", "0.45")),
                        help='discard masks covering more than this image fraction')
    parser.add_argument('--semantic_boost_max_detections_per_category', type=int,
                        default=int(os.environ.get("SEMANTIC_BOOST_MAX_DETECTIONS_PER_CATEGORY", "3")),
                        help='maximum detected boxes to segment per category per frame')
                                
    # for sem exp
    parser.add_argument('--lr', type=float, default=2.5e-5,
                        help='learning rate (default: 2.5e-5)')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=str, default=2,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--use_recurrent_global', type=int, default=0,
                        help='use a recurrent global policy')
    parser.add_argument('--reward_coeff', type=float, default=0.1,
                        help="Object goal reward coefficient")
    parser.add_argument('--intrinsic_rew_coeff', type=float, default=0.05,
                        help="intrinsic exploration reward coefficient")

    parser.add_argument('--load', type=str, default="0",
                    help="""model path to load,
                            0 to not reload (default: 0)""")
    # parse arguments
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if "objectnav_mp3d" in args.task_config and args.num_sem_categories == 16:
        args.num_sem_categories = 21

    return args
