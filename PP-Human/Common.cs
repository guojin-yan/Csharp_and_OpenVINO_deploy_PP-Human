using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCvSharp;

namespace PP_Human
{
    public class KeyPoint
    {
        // 视频帧
        public int mot_id { get; set; }
        // 关键点和分数
        public float[,] points { get; set; }
        // 行人位置框
        public Rect bbox { get; set; }

        public KeyPoint(int mot_id, float[,] points, Rect bbox)
        {
            this.mot_id = mot_id;
            this.points = points;
            this.bbox = bbox;
        }
    }
}
