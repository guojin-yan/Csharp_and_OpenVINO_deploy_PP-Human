using OpenVinoSharp;
using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PP_Human;

namespace ModelTimeTest
{
    internal class PP_YOLOE
    {
        // 成员变量
        string input_node_name = "image"; // 模型输入节点名称
        string output_node_name_1 = "tmp_20"; // 模型预测框输出节点名
        string output_node_name_2 = "concat_14.tmp_0"; // 模型预测置信值输出节点
        Size input_size = new Size(640, 640); // 模型输入节点形状
        int output_length = 8400; // 模型输出数据长度

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
            Console.WriteLine("行人识别：");
            Console.WriteLine("模型加载运行时间：{0} 毫秒", times[0] / n);
            Console.WriteLine("数据加载运行时间：{0} 毫秒", times[1] / n);
            Console.WriteLine("模型推理运行时间：{0} 毫秒", times[2] / n);
            Console.WriteLine("结果处理运行时间：{0} 毫秒", times[3] / n);
        }

        double[] yoloe_predict()
        {

            double[] times = new double[4];

            // 测试图片
            string image_path = @"E:\Git_space\基于Csharp和OpenVINO部署PP-Human\demo\hrnet_demo.jpg";
            Mat image = Cv2.ImRead(image_path);
            string mode_path = @"E:\Text_Model\PP-Human\poloe\paddle1\model.pdmodel";
            //string mode_path = @"E:\Text_Model\PP-Human\poloe\model.onnx"; // 目标检测模型
            //string mode_path = @"E:\Text_Model\PP-Human\poloe\ir\model.xml";
            //string mode_path = @"E:\Text_Model\PP-Human\poloe\ir_fp16\model.xml";



            // 加载模型
            DateTime begin = DateTime.Now;

            Core predictor = new Core(mode_path, "AUTO"); // 模型推理器

            DateTime end = DateTime.Now;
            TimeSpan oTime = end.Subtract(begin); //求时间差的函数  
            times[0] = oTime.TotalMilliseconds;


            // 加载输入数据
            begin = DateTime.Now;
            // 设置图片输入
            // 图片数据解码
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            // 设置图片输入
            predictor.load_input_data(input_node_name, input_image_data, input_image_length, 2);
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
            // 2125 765
            // 读取置信值结果
            float[] results_con = predictor.read_infer_result<float>(output_node_name_2, output_length);
            // 读取预测框
            float[] result_box = predictor.read_infer_result<float>(output_node_name_1, 4 * output_length);
            // 处理模型推理数据
            // 求取缩放大小
            double scale_x = (double)image.Width / (double)input_size.Width;
            double scale_y = (double)image.Height / (double)input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);
            ResBboxs result = process_result(results_con, result_box, scale_factor);

            end = DateTime.Now;
            oTime = end.Subtract(begin); //求时间差的函数  
            //Console.WriteLine("结果处理运行时间：{0} 毫秒", oTime.TotalMilliseconds);
            times[3] = oTime.TotalMilliseconds;
            predictor.delet();

            return times;
        }


        private ResBboxs process_result(float[] results_con, float[] result_box, Point2d scale_factor)
        {
            // 处理预测结果
            List<float> confidences = new List<float>();
            List<Rect> boxes = new List<Rect>();
            for (int c = 0; c < output_length; c++)
            {   // 重新构建
                Rect rect = new Rect((int)(result_box[4 * c] * scale_factor.X), (int)(result_box[4 * c + 1] * scale_factor.Y),
                    (int)((result_box[4 * c + 2] - result_box[4 * c]) * scale_factor.X),
                    (int)((result_box[4 * c + 3] - result_box[4 * c + 1]) * scale_factor.Y));
                boxes.Add(rect);
                confidences.Add(results_con[c]);
            }
            // 非极大值抑制获取结果候选框
            int[] indexes = new int[boxes.Count];
            CvDnn.NMSBoxes(boxes, confidences, 0.5f, 0.5f, out indexes);
            // 提取合格的结果
            List<Rect> boxes_result = new List<Rect>();
            List<float> con_result = new List<float>();
            List<int> clas_result = new List<int>();
            for (int i = 0; i < indexes.Length; i++)
            {
                boxes_result.Add(boxes[indexes[i]]);
                con_result.Add(confidences[indexes[i]]);
            }
            return new ResBboxs(boxes_result, con_result.ToArray(), clas_result.ToArray());
        }



    }
}
