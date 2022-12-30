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
        private Size2f coord_size = new Size2f(384, 512);
        private int input_length = 1700; // 模型输入节点形状
        private int output_length = 2; // 模型输出数据长度

        public STGCN(string mode_path, string device_name)
        {
            predictor = new Core(mode_path, device_name);
        }



        public KeyValuePair<string, float> predict(List<KeyPoints> points)
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
            KeyValuePair<string, float> result;
            if (results[0] > results[1])
            {
                result = new KeyValuePair<string, float>("falling", results[0]);
            }
            else 
            {
                result = new KeyValuePair<string, float>("unfalling", results[1]);
            }
            return result;
        }

        float[] preprocess_keypoint(List<KeyPoints> data) 
        { 
            float[] input_data = new float[this.input_length];

            for (int f = 0; f < 50; f++)
            {
                float[,] point = data[f].points;
                Rect rect = data[f].bbox;
                //Console.WriteLine(rect);
                for (int g = 0; g < 17; g++)
                {
                    //Console.WriteLine(point[g, 0]);
                    input_data[0 * 50 * 17 + f * 17 + g] = point[g, 0] / rect.Width * coord_size.Width;
                    input_data[1 * 50 * 17 + f * 17 + g] = point[g, 1] / rect.Height * coord_size.Height;
                }
            }

  

            for (int g = 0; g < 3; g++) 
            {
                Console.WriteLine(input_data[g]);
            }

            return input_data;
        }

        public void draw_result(ref Mat image, KeyValuePair<string, float> result, Rect rect)
        {
            Cv2.PutText(image, result.Key + ": " + result.Value.ToString(), new Point(rect.X, rect.Y + rect.Height + 10),
                HersheyFonts.HersheySimplex, 0.5, new Scalar(0, 0, 255));
        }
        public void release()
        {
            predictor.delet();
        }
    }
}
