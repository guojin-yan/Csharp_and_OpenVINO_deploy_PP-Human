using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PP_Human;
using OpenCvSharp;

namespace ProjectTest
{
    /// <summary>
    /// 测试人体摔倒检测
    /// </summary>
    public class HumanFallDown
    {

        /// <summary>
        /// 人体摔倒检测
        /// </summary>
        static public void human_fall_down()
        {

            // 模型路径  避免中文路径

            // ONNX格式
            string yoloe_path = @"E:\Text_Model\PP-Human\poloe\model.onnx"; // 目标检测模型
            string tinypose_path = @"E:\Text_Model\PP-Human\tinypose\model.onnx"; // 关键点检测模型
            // string tinypose_path = @"E:\Text_Model\TinyPose\tinypose_256_192\tinypose_256_192.onnx";
            string STGCN_path = @"E:\Text_Model\PP-Human\STGCN\model.onnx"; // 摔倒检测模型


            // 测试视频信息

            // 视频路径
            string test_video = @"E:\Git_space\基于Csharp和OpenVINO部署PP-Human\demo\摔倒.mp4";
            // string test_video = @"E:\Git_space\基于Csharp和OpenVINO部署PP-Human\demo\摔倒2.mp4";
            // 视频读取器
            VideoCapture video_capture = new VideoCapture(test_video);
            // 视频帧率
            double fps = video_capture.Fps;
            // 视频帧数
            int frame_count = video_capture.FrameCount;

            Console.WriteLine("video fps: {0}, frame_count: {1}", Math.Round(fps), frame_count);

            // 定义模型预测器
            // yoloe 模型预测
            YOLOE yoloe_predictor = new YOLOE(yoloe_path, "AUTO");
            Console.WriteLine("目标检测模型加载成功！！");
            // tinypose 预测器
            TinyPose tinyPose_predictor = new TinyPose(tinypose_path, "AUTO");
            Console.WriteLine("关键点识别模型加载成功！！");
            // STGCN 模型
            STGCN stgcn_predictor = new STGCN(STGCN_path, "AUTO");
            Console.WriteLine("摔倒识别模型加载成功！！");

            // 定义相关变量
            // 视频帧
            int frame_id = 0;
            // 视频帧图像
            Mat frame = new Mat();
            // 可视化
            Mat visualize_frame = new Mat();

            // 可视化窗口
            Window window = new Window("image");

            List<float[,]> points = new List<float[,]>();

            while (true) 
            {
                if (!video_capture.IsOpened()) 
                {
                    Console.WriteLine("视频打开失败！！");
                    break;
                }

                if (frame_id % 10 == 0)
                {
                    Console.WriteLine("检测进程 frame id: {0} - {1}", frame_id, frame_id + 10);
                }

                // 读取视频帧
                if (!video_capture.Read(frame)) 
                {
                    Console.WriteLine("视频读取完毕！！{0}",frame_id);
                    break;
                }

                frame_id++; // 帧号累加
                visualize_frame = frame.Clone();

                // 行人检测
                List<Rect> person_rects = yoloe_predictor.predict(frame);
                yoloe_predictor.draw_boxes(person_rects, ref visualize_frame);
                // 判断是否识别到人
                if (person_rects.Count < 1)
                {
                    continue;
                }

                // 裁剪行人区域
                Rect[] person_rect = person_rects.ToArray();
                Mat[] person_roi = cut_image_roi(frame, person_rect);

                // 关键点识别
                float[,] person_point = tinyPose_predictor.predict(person_roi[0]);
                //Console.WriteLine("{0}   {1}", person_point.GetLength(0), person_point.GetLength(1));
                points.Add(person_point);

                if (points.Count == 50) 
                {
                    stgcn_predictor.predict(points);
                    points.Clear();
                }
                
                for (int i = 0; i < 17; i++)
                {
                    person_point[i, 0] = person_point[i, 0] + person_rect[0].X;
                    person_point[i, 1] = person_point[i, 1] + person_rect[0].Y;
                }

                tinyPose_predictor.draw_poses(person_point, ref visualize_frame);
                window.ShowImage(visualize_frame);
                Cv2.WaitKey(1);

            }
        }

        /// <summary>
        /// 裁剪识别区域
        /// </summary>
        /// <param name="source_image">原图片</param>
        /// <param name="rects">矩形区域数组</param>
        /// <returns>文字区域mat</returns>
        public static Mat[] cut_image_roi(Mat source_image, Rect[] rects)
        {
            Mat image = source_image.Clone();
            Mat[] rois = new Mat[rects.Length];
            Rect sourse_rect = new Rect(0, 0, source_image.Cols, source_image.Rows);

            for (int r = 0; r < rects.Length; r++)
            {
                Point locate = rects[r].Location;
                int width = rects[r].Width;
                int height = rects[r].Height;
                double centre_X = locate.X + width * 0.5;
                double centre_Y = locate.Y + height * 0.5;
                double nwidth = width*1.3;
                double nheight = height * 1.3;
                Rect rect_new = new Rect((int)(centre_X - 0.5 * nwidth), (int)(centre_Y - 0.5 * nheight), (int)nwidth, (int)nheight);

                int x_min = Math.Max(0, rect_new.X);
                int x_max = Math.Min(source_image.Cols - 1, rect_new.X + rect_new.Width);
                int y_min = Math.Max(0, rect_new.Y);
                int y_max = Math.Min(source_image.Rows - 1, rect_new.Y + rect_new.Height);

                Rect rect = new Rect(x_min, y_min, x_max-x_min, y_max-y_min);

                //Rect rect = get_IOU(sourse_rect, rect_new);
                Mat roi = new Mat(image, rect);
                rois[r] = roi;
                rects[r] = rect;
            }
            return rois;

        }
    }
}
