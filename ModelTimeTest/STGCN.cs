using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenVinoSharp;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using PP_Human;

namespace ModelTimeTest
{
    internal class STGCN
    {
        // 成员变量
        private Core predictor; // 模型推理器
        private string input_node_name = "data_batch_0"; // 模型输入节点名称
        private string output_node_name = "reshape2_34.tmp_0"; // 模型预测输出节点名
        private Size2f coord_size = new Size2f(512, 384);
        private int input_length = 1700; // 模型输入节点形状
        private int output_length = 2; // 模型输出数据长度

        public void test_time()
        {
            int n = 100;
            double[] times = new double[4];
            for (int i = 0; i < n; i++)
            {
                double[] time = yoloe_predict();
                times[0] += time[0];
                times[1] += time[1];
                times[2] += time[2];
                times[3] += time[3];

            }
            Console.WriteLine("行为识别：");
            Console.WriteLine("模型加载运行时间：{0} 毫秒", times[0] / n);
            Console.WriteLine("数据加载运行时间：{0} 毫秒", times[1] / n);
            Console.WriteLine("模型推理运行时间：{0} 毫秒", times[2] / n);
            Console.WriteLine("结果处理运行时间：{0} 毫秒", times[3] / n);
        }

        double[] yoloe_predict()
        {

            double[] times = new double[4];


            //string mode_path = @"E:\Text_Model\PP-Human\STGCN\padddle\model.pdmodel";
            //string mode_path = @"E:\Text_Model\PP-Human\STGCN\model.onnx"; ; // 目标检测模型
            //string mode_path = @"E:\Text_Model\PP-Human\STGCN\ir\model.xml";
            string mode_path = @"E:\Text_Model\PP-Human\STGCN\ir_fp16\model.xml";

            // 加载模型
            DateTime begin = DateTime.Now;

            Core predictor = new Core(mode_path, "AUTO"); // 模型推理器

            DateTime end = DateTime.Now;
            TimeSpan oTime = end.Subtract(begin); //求时间差的函数  
            times[0] = oTime.TotalMilliseconds;


            // 加载输入数据
            begin = DateTime.Now;
            // 转换数据格式
            float[] input_data = preprocess_keypoint();
            // 设置模型输入
            predictor.load_input_data(input_node_name, input_data);
            end = DateTime.Now;
            //输出运行时间。  
            oTime = end.Subtract(begin); //求时间差的函数  
            // Console.WriteLine("数据加载运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            times[1] = oTime.TotalMilliseconds;


            begin = DateTime.Now;
            // 模型推理
            predictor.infer();
            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            //Console.WriteLine("模型推理运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            times[2] = oTime.TotalMilliseconds;
            begin = DateTime.Now;
            // 读取模型输出
            // 读取推理结果
            float[] results = predictor.read_infer_result<float>(output_node_name, output_length);

            KeyValuePair<string, float> result;
            if (results[0] > results[1])
            {
                result = new KeyValuePair<string, float>("falling", results[0]);
            }
            else
            {
                result = new KeyValuePair<string, float>("unfalling", results[1]);
            }

            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            //Console.WriteLine("结果处理运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            times[3] = oTime.TotalMilliseconds;
            predictor.delet();

            return times;
        }


        float[] preprocess_keypoint()
        {
            float[] input_data = new float[this.input_length];
            // (50, 17, 2)->(2, 50, 17)
            for (int f = 0; f < 50; f++)
            {
                for (int g = 0; g < 17; g++)
                {
                    input_data[1 * 50 * 17 + f * 17 + g] = 1;
                    input_data[0 * 50 * 17 + f * 17 + g] = 1;
                }
            }
            return input_data;
        }
    }
}
