
if __name__ == '__main__':
    # 创建解析器对象
    parser = argparse.ArgumentParser(description="Description of your script")
    
    # 添加命令行参数
    parser.add_argument("-alpha", type=float,default=0.1, help="Description of param1")
    parser.add_argument("-dname", type=str,default="fmnist", help="Description of param2")