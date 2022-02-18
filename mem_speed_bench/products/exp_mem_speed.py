import argparse
import json
import os
import time


def run_cmd(cmd):
    print(cmd)
    return os.system(cmd)


def to_command(conf_path, batch_size, swap, n_bits, not_compress_act, amp):
    cmd = f"python ./products/train_mini_batch.py --conf {conf_path} --batch_size {batch_size} --debug_mem --gpu 0"
    if not_compress_act:
        cmd += " --not_compress_act"
    if swap:
        cmd += " --swap"
    if amp:
        cmd += " --amp"
    cmd += f" --n_bits {n_bits}"
    return cmd


def round_up(x):
    return int((x + 9)// 10 * 10)

def round_down(x):
    return int(x // 10 * 10)


def run_benchmark(conf_path, batch_size, swap, n_bits, not_compress_act, amp):
    cmd = to_command(conf_path, batch_size, swap, n_bits, not_compress_act, amp)
    ret_code = run_cmd(cmd)

    # if ret_code != 0:
    #     out_file = "speed_results.json"
    #     with open(out_file, "a") as fout:
    #         val_dict = {
    #             "conf_path": conf_path,
    #             "swap": swap,
    #             "batch_size": batch_size,
    #         }
    #         fout.write(json.dumps(val_dict) + "\n")
    #     print(f"save results to {out_file}")

    time.sleep(1)
    run_cmd("nvidia-smi > /dev/null")
    time.sleep(1)
    return ret_code


def binary_search_max_batch(conf_path, swap, n_bits, not_compress_act, amp, low, high):
    ret = 0
    low, high= round_up(low), round_down(high)

    while low <= high:
        mid = round_down(low + (high - low) // 2)
        success = run_benchmark(conf_path, mid, swap, n_bits, not_compress_act, amp) == 0
        if success:
            ret = mid
            low = round_up(mid + 1)
        else:
            high = round_down(mid - 1)

    return ret



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str)
    parser.add_argument("--mode", type=str, default='binary_search_max_batch')
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()
    if args.mode == 'binary_search_max_batch':
        out_file = "max_batch_results.json"
        low, high = 200000, 1500000
        not_compress_act = True
        max_batch_size = binary_search_max_batch(args.conf, False, 8, True, False, low, high)
        with open(out_file, "a") as fout:
            val_dict = {
                "conf": args.conf,
                "not_compress_act": not_compress_act,
                "max_batch_size": max_batch_size,
                "tstamp": time.time()
            }
            fout.write(json.dumps(val_dict) + "\n")
            print('finished vanilla result')

        for n_bits in [1, 2, 4, 8]:
            for swap in [True, False]:
                for amp in [True, False]:
                    not_compress_act = False
                    max_batch_size = binary_search_max_batch(args.conf, swap, n_bits, not_compress_act, amp, low, high)
                    with open(out_file, "a") as fout:
                        val_dict = {
                            "conf": args.conf,
                            "n_bits": n_bits,
                            "swap": swap,
                            "amp": amp,
                            "max_batch_size": max_batch_size,
                            "tstamp": time.time()
                        }
                        fout.write(json.dumps(val_dict) + "\n")
                    print(f"save results to {out_file}")
    else:
        raise ValueError()
