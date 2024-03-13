def rpb2rpc_lzj(f_in_name, f_out_name, transform_type):
    if transform_type == 1:
        # 1 读取 RPB 文件内容
        with open(f_in_name, 'r') as fid1:
            lines = fid1.readlines()
        lines = [line.strip() for line in lines][4:]  # 去除头部无用行
        
        # 使用字典存储参数
        params = {}
        keys = [
            'errBias', 'errRand', 'lineOffset', 'sampOffset', 'latOffset',
            'longOffset', 'heightOffset', 'lineScale', 'sampScale',
            'latScale', 'longScale', 'heightScale'
        ]
        values = [line.split('=')[1].split(';')[0].strip() for line in lines[:12]]
        for key, value in zip(keys, values):
            params[key] = value
        
        coef_keys = ['lineNumCoef', 'lineDenCoef', 'sampNumCoef', 'sampDenCoef']
        for key in coef_keys:
            params[key] = []
            for i in range(20):
                value = lines[12].split('=')[1].strip(' ,();')
                params[key].append(value)
                lines.pop(0)
        
        # 2 将 RPB 文件写为 RPC 文件
        with open(f_out_name, 'w') as fid2:
            for key in ['lineOffset', 'sampOffset', 'latOffset', 'longOffset',
                        'heightOffset', 'lineScale', 'sampScale', 'latScale',
                        'longScale', 'heightScale']:
                fid2.write(f'{key.upper()}: {params[key]}\n')
            
            for key in coef_keys:
                for i, value in enumerate(params[key], 1):
                    fid2.write(f'{key.upper()}_{i}: {value}\n')

    elif transform_type == 2:
        # 1 读取 RPC 文件内容
        with open(f_in_name, 'r') as fid1:
            lines = fid1.readlines()
        lines = [line.strip() for line in lines]
        
        # 使用字典存储参数
        params = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':')
                key = key.strip().replace(' ', '').lower()
                value = value.strip().split()[0]  # 去除单位
                params[key] = value
        
        # 2 将 RPC 文件写为 RPB 文件
        with open(f_out_name, 'w') as fid2:
            fid2.write('satId = "XXX";\nbandId = "XXX";\nSpecId = "XXX";\nBEGIN_GROUP = IMAGE\n')
            for key in ['errBias', 'errRand', 'lineOffset', 'sampOffset', 'latOffset',
                        'longOffset', 'heightOffset', 'lineScale', 'sampScale',
                        'latScale', 'longScale', 'heightScale']:
                fid2.write(f'\t{key} =   {params[key]};\n')
            
            for key in ['lineNumCoef', 'lineDenCoef', 'sampNumCoef', 'sampDenCoef']:
                fid2.write(f'\t{key} = (\n')
                for i in range(1, 21):
                    value = params[f'{key}_{i}']
                    if i < 20:
                        fid2.write(f'\t\t\t{value},\n')
                    else:
                        fid2.write(f'\t\t\t{value});\n')
            fid2.write('END_GROUP = IMAGE\nEND;')

#新函数，将rpb文件转换成rpc文件
def extract_coefficients(text, key):
    start_idx = text.find(f"{key} = (") + len(f"{key} = (")
    end_idx = text.find(");", start_idx)
    coefficients = text[start_idx:end_idx]
    return [x.strip() for x in coefficients.strip().split(',')]

def rpb2rpc(f_in_name, f_out_name):
    # 1 读取 RPB 文件内容
    with open(f_in_name, 'r') as fid1:
        content = fid1.read()

    params = {}
    
    # 提取普通参数
    #rpb参数
    keys = [
        'errBias', 'errRand', 'lineOffset', 'sampOffset', 'latOffset',
        'longOffset', 'heightOffset', 'lineScale', 'sampScale',
        'latScale', 'longScale', 'heightScale'
    ]
     #rpc参数
    rpc_keys = [
        'ERRBIAS', 'ERRRAND', 'LINE_OFF', 'SAMP_OFF', 'LAT_OFF',
        'LONG_OFF', 'HEIGHT_OFF', 'LINE_SCALE', 'SAMP_SCALE',
        'LAT_SCALE', 'LONG_SCALE', 'HEIGHT_SCALE'
    ]
    for key in keys:
        start_idx = content.find(f"{key} = ") + len(f"{key} = ")
        end_idx = content.find(";", start_idx)
        params[key] = content[start_idx:end_idx].strip()

    # 提取系数
    coef_keys = ['lineNumCoef', 'lineDenCoef', 'sampNumCoef', 'sampDenCoef']
    # rpc_coef_keys = ['LINE_Num_COEFF', 'LINE_DEN_COEFF', 'SAMP_NUM_COEFF', 'SAMP_DEN_COEFF']
    for key in coef_keys:
        params[key] = extract_coefficients(content, key)

    # 格式化 RPC 文件中的系数关键字
    def format_rpc_coef_key(key):
        key = key.replace('Num', 'NUM').replace('Den', 'DEN').upper()  # 修改处
        key = key.replace('COEF', '_COEFF')  # 进一步修改以确保正确格式L
        key = key.replace('SAMP', 'SAMP_')  # 进一步修改以确保正确格式L
        return key.replace('LINE', 'LINE_')

    # 2 写出 RPC 文件
    with open(f_out_name, 'w') as fid2:
        # 写普通参数
        i=0
        for key in keys: 
            #fid2.write(f'{key.upper()}: {params[key]}\n')
            fid2.write(f'{rpc_keys[i]}: {params[key]}\n')
            i+=1
        # 写系数参数
        for key in coef_keys:
            coef_params = params[key]
            rpc_key = format_rpc_coef_key(key)
            for i, coef in enumerate(coef_params, start=1):
                fid2.write(f'{rpc_key}_{i}: {coef}\n')

