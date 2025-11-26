from train.log_utils import log_info

if __name__ == "__main__":
    with open("../autodl-tmp/UDA/distanDANN/log.txt", "a") as file:
        file.write("hello" + '\n')
        print("content")