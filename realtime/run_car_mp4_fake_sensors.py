import os
import sys

from radartracks_processing_person_car_cpu_debug import main as fusion_main


def pop_flag(args, flag):
    if flag not in args:
        return False
    args.remove(flag)
    return True


def option_present(args, option):
    return any(arg == option or arg.startswith(option + "=") for arg in args)


def main():
    user_args = sys.argv[1:]
    once = pop_flag(user_args, "--once")
    record = pop_flag(user_args, "--record")
    headless = pop_flag(user_args, "--headless")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    forwarded = ["--offline-demo"]

    if not option_present(user_args, "--input"):
        forwarded.extend(["--input", os.path.join(script_dir, "car.mp4")])
    if not option_present(user_args, "--realtime-playback"):
        forwarded.append("--realtime-playback")
    if not once and not option_present(user_args, "--loop"):
        forwarded.append("--loop")
    if not headless and "--show" not in user_args:
        forwarded.append("--show")
    if not record and "--no-output" not in user_args:
        forwarded.append("--no-output")

    fusion_main(forwarded + user_args)


if __name__ == "__main__":
    main()
