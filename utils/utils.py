import yaml

# ------------------------------- Printing helpers --------------------------------
def print_header(title, width=60):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_subheader(title, width=60):
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)

def print_info(label, value, indent=2):
    spaces = " " * indent
    print(f"{spaces}{label}: {value}")

def print_success(message, indent=2):
    spaces = " " * indent
    print(f"{spaces}✓ {message}")

def print_progress(step, total_steps, description):
    progress = f"[{step}/{total_steps}]"
    print(f"\n{progress} {description}")

def print_config_block(title: str, **items):
    width = max(len(k) for k in items) if items else 0
    line = "=" * 120
    print("\n" + line)
    print(f"{title}".center(120))
    print("-" * 120)
    for k, v in items.items():
        print(f"{k:<{width}} : {v}")
    print(line + "\n")

# ------------------------------- YAML update -------------------------------------
def update_yaml_video_path(yaml_file_path, new_video_path, out_dir, new_dim, fps, num_frames):
    """
    Update the video_path and related fields in the YAML configuration file.
    """
    try:
        with open(yaml_file_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        config['video_path'] = new_video_path
        config['save_path'] = out_dir
        config['fps'] = fps
        config['width'] = new_dim[0]
        config['height'] = new_dim[1]
        config['num_frames'] = num_frames
        print(f'Updated video_path in YAML to: {new_video_path}')
        with open(yaml_file_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        print(f'Successfully updated {yaml_file_path}')
    except Exception as e:
        print(f'Error updating YAML file: {e}')
        raise
