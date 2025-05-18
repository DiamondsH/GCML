# 定义一个函数来处理文件
def process_file(input_file_path, output_file_path):
    try:
        # 打开输入文件进行读取
        with open(input_file_path, 'r') as file:
            lines = file.readlines()

        # 处理每一行，删除第二个'=='及其后面的内容
        processed_lines = []
        for line in lines:
            # 分割行，获取第二个'=='之前的内容
            parts = line.split('==')
            if len(parts) > 1:
                # 只保留第一个'=='之前的内容
                processed_line = parts[0] + '==' +parts[1]
                processed_lines.append(processed_line)
            else:
                # 如果没有'=='，则保留原行
                processed_lines.append(line.strip())

        # 将处理后的内容写入新文件
        with open(output_file_path, 'w') as file:
            for processed_line in processed_lines:
                file.write(processed_line + '\n')

        print(f"File processed successfully and saved to {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


# 调用函数，传入输入文件和输出文件的路径
input_file_path = 'requirements.txt'  # 输入文件的路径
output_file_path = 'output.txt'  # 输出文件的路径
process_file(input_file_path, output_file_path)