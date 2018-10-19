using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore.Data
{
    public class Line
    {
        private Point _startPoint;
        private Point _endPoint;

        public Point StartPoint { get => _startPoint; set => _startPoint = value; }
        public Point EndPoint { get => _endPoint; set => _endPoint = value; }

        public Line(Point startPoint, Point endPoint)
        {
            StartPoint = startPoint;
            EndPoint = endPoint;
        }
        public Line() { }
    }

    public class VLine : Line
    {
        public VLine(Point startPoint, Point endPoint) : base(startPoint, endPoint) { }

        public VLine(int coord, Size size)
        {
            StartPoint = new Point(coord, 0);
            EndPoint = new Point(coord, size.Height);
        }
    }

    public class HLine : Line
    {
        public HLine(Point startPoint, Point endPoint) : base(startPoint, endPoint) { }

        public HLine(int coord, Size size)
        {
            StartPoint = new Point(0, coord);
            EndPoint = new Point(size.Width, coord);
        }
    }

    public struct CropLine
    {
        internal int coord;
        internal bool hard;

        public CropLine(int coord, bool hard = false)
        {
            this.coord = coord;
            this.hard = hard;
        }

        public override string ToString()
        {
            return $"{coord} - {hard}";
        }
    }

    public struct RotationData
    {
        internal double angle;
        internal LineSegment2D[] lines;

        public RotationData(double angle, LineSegment2D[] lines)
        {
            this.angle = angle;
            this.lines = lines;
        }

        public override string ToString()
        {
            return $"{angle} - [{lines.Length}]";
        }
    }

    public struct IndexValuePair
    {
        internal int index;
        internal int value;

        public IndexValuePair(int index, int value)
        {
            this.index = index;
            this.value = value;
        }

        public override string ToString()
        {
            return $"{value} - [{index}]";
        }
    }

    public struct IndexValuePairF
    {
        internal int index;
        internal float value;

        public IndexValuePairF(int index, float value)
        {
            this.index = index;
            this.value = value;
        }

        public IndexValuePairF(int index, double value)
        {
            this.index = index;
            this.value = (float)value;
        }

        public override string ToString()
        {
            return $"{value} - [{index}]";
        }
    }
}