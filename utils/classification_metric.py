import json
import os
import codecs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_score, classification_report
import warnings

warnings.filterwarnings('ignore')


class ClassificationMetric:
    def __init__(self, labels, save_path, text_flag=1, cmap=plt.cm.Blues, show_pic=False, save_result=True):
        """

        :param labels: 所有的类别名称；类型为list；维度为[n_classes]
        :param save_path: 保存混淆矩阵与log文件的路径；类型为str
        :param text_flag: text_flag=0标记在图片中对文字是正确个数，text_flag=1标记在图片中对文字是正确率，text_flag=2没有数字；类型为int
        :param cmap: plt中的cmap
        :param show_pic: 是否显示画出的混淆矩阵；类型为bool
        :param save_result: 是否保存画出的混淆矩阵，以及是否存放结果到log文件；类型为bool
        """
        self.labels = labels
        self.save_path = save_path
        self.text_flag = text_flag
        self.cmap = cmap
        self.show_pic = show_pic
        self.save_result = save_result

    def get_metric(self, y_true, y_pred):
        """

        :param y_true: 预测出的类标；类型为numpy；维度为[n_samples]
        :param y_pred: 真实类标；类型为numpy；维度为[n_samples]
        :return classify_report: 分类报告；类型为string / dict
        :return my_confusion_matrix: 混淆矩阵；类型为numpy；维度为[n_classes, n_classes]
        :return acc_for_each_class: 每一类的精度；类型为numpy；维度为[n_classes]
        :return oa: 微平均（是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标）；类型为float
        :return average_accuracy: 宏平均（先对每一个类统计指标值，然后在对所有类求算术平均值）；类型为float
        :return kappa: Kappa系数；类型为float
        """

        # 得到分类报告
        classify_report = classification_report(y_true, y_pred)

        # 得到混淆矩阵
        my_confusion_matrix = confusion_matrix(y_true, y_pred)

        # 得到kappa，oa，aa以及各类的分类精度
        kappa = cohen_kappa_score(y_pred, y_true)
        oa = accuracy_score(y_true, y_pred)
        acc_for_each_class = precision_score(y_true, y_pred, average=None)
        average_accuracy = np.mean(acc_for_each_class)
        return classify_report, my_confusion_matrix, acc_for_each_class, oa, average_accuracy, kappa

    def draw_cm_and_save_result(self, classify_report, my_confusion_matrix, acc_for_each_class, oa, average_accuracy,
                                kappa):
        """ 画混淆矩阵并根据self.save_result的值决定是否保存最后结果

        :param classify_report: 分类报告；类型为string / dict
        :param my_confusion_matrix: 混淆矩阵；类型为numpy；维度为[n_classes, n_classes]
        :param acc_for_each_class: 每一类的精度；类型为numpy；维度为[n_classes]
        :param oa: 微平均（是对数据集中的每一个实例不分类别进行统计建立全局混淆矩阵，然后计算相应指标）；类型为float
        :param average_accuracy: 宏平均（先对每一个类统计指标值，然后在对所有类求算术平均值）；类型为float
        :param kappa: Kappa系数；类型为float
        """

        np.set_printoptions(precision=2)
        my_confusion_matrix_normalized = my_confusion_matrix.astype('float') / my_confusion_matrix.sum(axis=1) \
            [:, np.newaxis]
        plt.figure(figsize=(20, 16), dpi=120)

        ind_array = np.arange(len(self.labels))
        x, y = np.meshgrid(ind_array, ind_array)
        # intFlag = 0 # 标记在图片中对文字是整数型还是浮点型
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            if self.text_flag == 0:
                c = my_confusion_matrix[y_val][x_val]
                # 这里是绘制数字，可以对数字大小和颜色进行修改
                plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=8, va='center', ha='center')
                plt.grid(True, which='minor', linestyle='-')
            elif self.text_flag == 1:
                c = my_confusion_matrix_normalized[y_val][x_val]
                if c > 0.01:
                    plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
                else:
                    plt.text(x_val, y_val, "%d" % (0,), color='red', fontsize=7, va='center', ha='center')
                plt.grid(True, which='minor', linestyle='-')
            else:
                pass

        if self.text_flag == 0:
            plt.imshow(my_confusion_matrix, interpolation='nearest', cmap=self.cmap)
        else:
            plt.imshow(my_confusion_matrix_normalized, interpolation='nearest', cmap=self.cmap)

        tick_marks = np.array(range(len(self.labels))) + 0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')

        plt.gcf().subplots_adjust(bottom=0.15)
        plt.title('Confusion Matrix')
        plt.colorbar()
        xlocations = np.array(range(len(self.labels)))
        font = FontProperties(fname=r"font/simhei.ttf", size=7)
        plt.xticks(xlocations, self.labels, fontproperties=font, rotation=90)
        plt.yticks(xlocations, self.labels, fontproperties=font, rotation=0)
        plt.ylabel('Index of True Classes')
        plt.xlabel('Index of Predict Classes')

        if self.save_result:
            plt.savefig(os.path.join(self.save_path, 'confusion_matrix'), dpi=200)

            result = {'acc_for_each_class': acc_for_each_class.tolist(),
                      'OA': oa, 'AA': average_accuracy, 'kappa': kappa,
                      'classify_report': classify_report,
                      'my_confusion_matrix': my_confusion_matrix.tolist()}
            with codecs.open(os.path.join(self.save_path, 'result.json'), 'w', "utf-8") as json_file:
                json.dump(result, json_file, ensure_ascii=False)

            classes_acc = {}
            for label_name, each_class_acc in zip(self.labels, acc_for_each_class.tolist()):
                classes_acc[label_name] = each_class_acc
            with codecs.open(os.path.join(self.save_path, 'classes_acc.json'), 'w', "utf-8") as json_file:
                json.dump(classes_acc, json_file, ensure_ascii=False)

        if self.show_pic:
            plt.show()
        plt.close('all')
