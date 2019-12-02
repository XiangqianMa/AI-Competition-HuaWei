# Author: XiangqianMa
import imghdr
import os
import struct
import cv2

###################################################
# 功能：查找数据集中存储格式错误的图片，并将其转换为统一的格式
###################################################

# 数据集存放路径
image_path = "/media/mxq/data/competition/HuaWei/酥饺"
# 数据集路径下是否有子文件夹，有则设置该参数为True，否则为False
sub_dir_exit = False
# 目标格式
dir_format = "jpg"
type_dict = {
 
    'FFD8FF':'jpg','89504E47':'png','47494638':'gif','49492A00':'tif',
    '424D':'bmp','41433130':'dwg','38425053':'psd','7B5C727466':'rtf','3C3F786D6C':'xml',
    '68746D6C3E':'html','44656C69766572792D646174653A':'eml','CFAD12FEC5FD746F':'dbx','2142444E':'pst',
    'D0CF11E0':'doc/xls','5374616E64617264204A':'mdb','FF575043':'wpd','252150532D41646F6265':'ps/eps',
    '255044462D312E':'pdf','AC9EBD8F':'qdf','E3828596':'pwl','504B0304':'zip',
    '52617221':'rar','57415645':'wav','41564920':'avi','2E7261FD':'ram',
    '2E524D46':'rm','000001BA':'mpg','000001B3':'mpg','6D6F6F76':'mov','3026B2758E66CF11':'asf','4D546864':'mid'
}
 
#转成16进制字符串
def bytes2hex(bytes):
    num = len(bytes)
    hexstr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexstr += u"0"
        hexstr += t
    return hexstr.upper()
 
#获得类型
def get_filetype(filename):
    file = open(filename, 'rb')
    ftype = 'unknown'
    
    try:
        for k,v in type_dict.items():
            # 16进制每一位占4位
            num_bytes = int(len(k)/2)
            file.seek(0)
            # 一个byte占8位(一个字节)，故两个十六进制所占的位数相当于一个char，
            # 所以从文件中读取char时，对应的二进制位数需要减半
            hbytes = struct.unpack('B'*num_bytes, file.read(num_bytes))
            code = bytes2hex(hbytes)
            if code == k:
                ftype =  v
                break
        file.close()
        status = True
    except:
        status = False
    return ftype, False


def modify_image_formate(image_name, origin_format, format='.jpg'):
    '''修改图片为正确的存储格式
    origin_format:图片的正确格式
    image_name: 待修改的图片的存储路径
    format:　目标格式
    
    '''
    if origin_format == 'png' or origin_format == 'bmp':
        image = cv2.imread(image_name)
        dir_image_name = image_name.split('.')[0] + format

        os.remove(dir_image_name)
        cv2.imwrite(dir_image_name, image)
    elif origin_format == 'gif':
        gif = cv2.VideoCapture(image_name)
        success, frame = gif.read()
        while(success):
            dir_image_name = image_name.split('.')[0] + format

            os.remove(dir_image_name)
            cv2.imwrite(dir_image_name, frame)
            success, frame = gif.read()
        
        gif.release()


def clean_samll_size_file(size_thresh, data_root):
    """清理data_root下文件大小小于size_thresh的文件

    Args:
        size_thresh: 文件大小阈值， bytes
        data_root: 数据集根目录
    """
    files = os.listdir(data_root)
    for file_name in files:
        size = os.path.getsize(os.path.join(data_root, file_name))
        if size < size_thresh:
            print('Removing %s' % (os.path.join(data_root, file_name)))
            os.remove(os.path.join(data_root, file_name))
    

if __name__ == "__main__":
    images = os.listdir(image_path)

    for image in images:
        print("------------{}----------".format(image))
        image_name = os.path.join(image_path, image)
        image_type, status = get_filetype(image_name)

        # 图片存储格式正确时，跳过当前图片，否则修改图片存储格式
        if image_type is dir_format:
            continue
        elif image_type == 'unknown' or not status:
            print('Removing %s' % image_name)
            os.remove(image_name)
        else:
            print("Modifing {}, it's right format is: {}.".format(image_name, image_type))
            modify_image_formate(image_name, origin_format=image_type, format='.jpg')
    # 移除过小的文件
    clean_samll_size_file(1024, image_path)