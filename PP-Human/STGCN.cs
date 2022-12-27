using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;
using OpenVinoSharp;
using OpenCvSharp.Dnn;

namespace PP_Human
{
    public class STGCN
    {
        // 成员变量
        private Core predictor; // 模型推理器
        private string input_node_name = "data_batch_0"; // 模型输入节点名称
        private string output_node_name = "reshape2_34.tmp_0"; // 模型预测输出节点名
        private int input_length = 1700; // 模型输入节点形状
        private int output_length = 2; // 模型输出数据长度

        public STGCN(string mode_path, string device_name)
        {
            predictor = new Core(mode_path, device_name);
        }



        public bool predict(List<float[,]> points)
        {
            // 转换数据格式
            float[] input_data = preprocess_keypoint(points);
            // 设置模型输入
            predictor.load_input_data(input_node_name, input_data);
            // 模型推理
            predictor.infer();

            // 读取推理结果
            float[] results = predictor.read_infer_result<float>(output_node_name, output_length);

            Console.WriteLine("{0}   {1}", results[0], results[1]);
            return true;
        }

        float[] preprocess_keypoint(List<float[,]> data) 
        { 
            float[] input_data = new float[this.input_length];

            for (int p = 0; p < 2; p++) 
            {
                for (int f = 0; f < 50; f++) 
                {
                    for (int g = 0; g < 17; g++) 
                    {
                        input_data[p * 50 * 17 + f * 17 + g] = data[f][g, p];
                    }
                }
            }
            return input_data;
        }
    }
}
