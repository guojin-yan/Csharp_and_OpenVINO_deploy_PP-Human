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
    public class YOLOE
    {
        // 成员变量
        private Core predictor; // 模型推理器
        private string input_node_name = "image"; // 模型输入节点名称
        private string output_node_name_1 = "tmp_20"; // 模型预测框输出节点名
        private string output_node_name_2 = "concat_14.tmp_0"; // 模型预测置信值输出节点
        private Size input_size = new Size(640, 640); // 模型输入节点形状
        private int output_length = 8400; // 模型输出数据长度

        public YOLOE(string mode_path, string device_name)
        {
            predictor = new Core(mode_path, device_name);
        }



        public ResBboxs predict(Mat image)
        {
            // 设置图片输入
            // 图片数据解码
            byte[] input_image_data = image.ImEncode(".bmp");
            // 数据长度
            ulong input_image_length = Convert.ToUInt64(input_image_data.Length);
            // 设置图片输入
            predictor.load_input_data(input_node_name, input_image_data, input_image_length, 2);

            // 求取缩放大小
            double scale_x = (double)image.Width / (double)this.input_size.Width;
            double scale_y = (double)image.Height / (double)this.input_size.Height;
            Point2d scale_factor = new Point2d(scale_x, scale_y);
            // 模型推理
            predictor.infer();

            // 读取置信值结果
            float[] results_con = predictor.read_infer_result<float>(output_node_name_2, output_length);
            // 读取预测框
            float[] result_box = predictor.read_infer_result<float>(output_node_name_1, 4 * output_length);
            // 处理模型推理数据
            ResBboxs result = process_result(results_con, result_box, scale_factor);
            return result;
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
            return new ResBboxs(boxes_result, con_result.ToArray(),clas_result.ToArray());
        }

        public void draw_boxes(ResBboxs result, ref Mat image) 
        {
            for (int i = 0; i < result.bboxs.Count; i++) 
            {
                Cv2.Rectangle(image, result.bboxs[i], new Scalar(255, 0, 0), 3);
                Cv2.PutText(image, "score: " + result.scores[i].ToString(), new Point(result.bboxs[i].X, result.bboxs[i].Y - 10),
                    HersheyFonts.HersheySimplex, 0.6, new Scalar(0, 0, 255), 2);
            }
        }

        public void release() 
        {
            predictor.delet();
        }

    }
}