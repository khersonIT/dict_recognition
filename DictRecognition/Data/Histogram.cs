using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore.Data
{
    public static class Extentions
    {
        public static Histogram GetBlancHist(this IImage img)
        {
            return new Histogram(img.Size);
        }
        public static Histogram CalculateHist(this Image<Gray, byte> img)
        {
            var result = img.GetBlancHist();

            byte[,,] data = img.Data;

            for (int i = 0; i < img.Rows; i += 1)
            {
                bool bStart = false;
                bool hStart = false;
                for (int j = 0; j < img.Cols; j += 1)
                {
                    if (data[i, j, 0] != 0)
                    {
                        if (!hStart)
                            hStart = true;

                        result.hStats[i]++;
                        result.hlStats[i].data++;

                        if (!bStart)
                        {
                            if (result.hlStats[i].borders == null)
                                result.hlStats[i].borders = new List<Border>();

                            result.hlStats[i].borders.Add(new Border());
                            result.hlStats[i].borders.Last().start = j;

                            bStart = true;
                        }

                        result.vStats[j]++;
                        result.vlStats[j]++;
                    }
                    else
                    {
                        if (hStart)
                            result.hlStats[i].data--;

                        result.vlStats[j]--;

                        if (bStart)
                        {
                            result.hlStats[i].borders.Last().end = j;
                            bStart = false;
                        }
                    }
                }
            }

            result.hStats[0] = 0;
            result.hStats[result.hStats.Length - 1] = 0;

            result.vStats[0] = 0;
            result.vStats[result.vStats.Length - 1] = 0;

            result.hMax = result.hStats.Max();
            result.hlMax = result.hlStats.Max(x => x.data);

            result.vMax = result.vStats.Max();
            result.vlMax = result.vlStats.Max();

            return result;
        }
        public static void VisualizeHist(this Image<Bgr, byte> img, Histogram hist, bool horizontal = false, bool vertical = false, int dh = 240, int dv = 240, int border = 20, double threshold = .1d)
        {
            var sSize = new Size(img.Size.Width - dh, img.Size.Height - dv);

            if (horizontal)
            {
                for (int i = 0; i < hist.hStats.Length; i++)
                    if (hist.hStats[i] != 0)
                    {
                        var color = ((double)hist.hStats[i] / hist.hMax < threshold) ? new MCvScalar(255, 255, 255) : new MCvScalar(0, 0, 255);
                        CvInvoke.Line(img, new Point(border, dv + i), new Point(border + ((dh - 2 * border) * hist.hStats[i] / hist.hMax), dv + i), color);
                    }

                CvInvoke.Line(img, new Point(border, dv), new Point(border, sSize.Height + dv), new MCvScalar(255, 255, 255), 2);
                CvInvoke.Line(img, new Point(dh - border, dv), new Point(dh - border, sSize.Height + dv), new MCvScalar(255, 255, 255), 2);
            }

            if (vertical)
            {
                for (int i = 0; i < hist.vStats.Length; i++)
                    if (hist.vStats[i] != 0)
                    {
                        var color = ((double)hist.vStats[i] / hist.vMax < threshold) ? new MCvScalar(255, 255, 255) : new MCvScalar(0, 0, 255);
                        CvInvoke.Line(img, new Point(dh + i, border), new Point(dh + i, border + ((dv - 2 * border) * hist.vStats[i] / hist.vMax)), color);
                    }

                CvInvoke.Line(img, new Point(240, 20), new Point(sSize.Width + 240, 20), new MCvScalar(255, 255, 255), 2);
                CvInvoke.Line(img, new Point(240, 220), new Point(sSize.Width + 240, 220), new MCvScalar(255, 255, 255), 2);
            }
        }
        public static void VisualizeExtremums(this Image<Bgr, byte> img, List<IndexValuePair> extr, bool horizontal = false, bool vertical = false, int dh = 240, int dv = 240, int border = 20, double threshold = .5d)
        {
            var sSize = new Size(img.Size.Width - dh, img.Size.Height - dv);

            var maxExtr = extr.Max(x => x.value);
            if (horizontal)
            {
                for (int i = 0; i < extr.Count; i++)
                {
                    var color = ((double)extr[i].value / maxExtr) < threshold ? new MCvScalar(255, 255, 255) : new MCvScalar(0, 0, 255);
                    CvInvoke.Line(img, new Point(border, dv + extr[i].index), new Point(border + ((dh - 2 * border) * extr[i].value / maxExtr), dv + extr[i].index), color, 2);
                }

                CvInvoke.Line(img, new Point(border, dv), new Point(border, sSize.Height + dv), new MCvScalar(255, 255, 255), 2);
                CvInvoke.Line(img, new Point(dh - border, dv), new Point(dh - border, sSize.Height + dv), new MCvScalar(255, 255, 255), 2);
            }

            if (vertical)
            {
                for (int i = 0; i < extr.Count; i++)
                {
                    var color = ((double)extr[i].value / maxExtr) < threshold ? new MCvScalar(255, 255, 255) : new MCvScalar(0, 0, 255);
                    CvInvoke.Line(img, new Point(dh + extr[i].index, border), new Point(dh + extr[i].index, border + ((dv - 2 * border) * extr[i].value / maxExtr)), color, 2);
                }

                CvInvoke.Line(img, new Point(240, 20), new Point(sSize.Width + 240, 20), new MCvScalar(255, 255, 255), 2);
                CvInvoke.Line(img, new Point(240, 220), new Point(sSize.Width + 240, 220), new MCvScalar(255, 255, 255), 2);
            }
        }
        public static void VisualizeLines(this Image<Bgr, byte> img, int[] vLines, int[] hLines, int dh = 240, int dv = 240, MCvScalar? color = null)
        {
            if (color == null)
                color = new MCvScalar(0, 0, 255);

            if (vLines != null)
                foreach (var x in vLines)
                    CvInvoke.Line(img, new Point(dh + x, dv), new Point(dh + x, img.Size.Height), (MCvScalar)color, 3);

            if (hLines != null)
                foreach (var x in hLines)
                    CvInvoke.Line(img, new Point(dh, dv + x), new Point(dh + img.Size.Width, dv + x), (MCvScalar)color, 3);
        }
        public static void VisualizeStartPoints(this Image<Bgr, byte> img, int[] points, int average = -1, int dh = 240, int dv = 240, MCvScalar? color = null)
        {
            if (color == null)
                color = new MCvScalar(0, 0, 255);

            if (points != null)
                for (int i = 0; i < points.Length; i++)
                    CvInvoke.Line(img, new Point(dh + points[i], i + 1), new Point(dh + points[i] + 20, i + 1), (MCvScalar)color, 2);

            if (average != -1)
                CvInvoke.Line(img, new Point(dh + average, dv), new Point(dh + average, dv + img.Size.Height), new MCvScalar(255, 0, 0), 2);
        }
        public static void VisualizeSegments(this Image<Bgr, byte> img, PageSegment[] segments, int border = -1, int dh = 240, int dv = 240, MCvScalar? color1 = null, MCvScalar? color2 = null)
        {
            if (color1 == null)
                color1 = new MCvScalar(0, 0, 255);
            if (color2 == null)
                color2 = new MCvScalar(255, 0, 0);

            if (segments != null)
                foreach (var x in segments)
                {
                    if (x.start.X > border)
                        CvInvoke.Rectangle(img, x.ToRect(dh, dv), (MCvScalar)color1);
                    else
                        CvInvoke.Rectangle(img, x.ToRect(dh, dv), (MCvScalar)color2);
                }
        }
        public static void VisualizeLines(this Image<Gray, byte> img, LineSegment2D[] lines)
        {
            foreach (var l in lines)
                img.Draw(l, new Gray(255), 1);
        }
    }

    public class Histogram
    {
        internal int[] hStats;
        internal LHist[] hlStats;
        internal int[] vStats;
        internal int[] vlStats;

        internal int hMax;
        internal int hlMax;
        internal int vMax;
        internal int vlMax;

        internal int hlStart;
        internal int hlEnd;
        internal int vlStart;
        internal int vlEnd;

        public Histogram(Size imgSize)
        {
            hStats = new int[imgSize.Height];
            hlStats = new LHist[imgSize.Height];

            vStats = new int[imgSize.Width];
            vlStats = new int[imgSize.Width];

            hMax = -1;
            hlMax = -1;
            vMax = -1;
            vlMax = -1;

            hlStart = -1;
            hlEnd = -1;
            vlStart = -1;
            vlEnd = -1;
        }

        public void NegativeZeroTransform(bool horizontal = false, bool vertical = false)
        {
            if (horizontal)
                for (int i = 0; i < hlStats.Length; i++)
                    if (hlStats[i].data < 0)
                        hlStats[i].data = -1;
        }

        public List<CropLine> CalculateVerticalBorders(double threshold, bool inverted = false)
        {
            var result = new List<CropLine>();

            int step = 0;

            var predt = inverted ? new Predicate<double>(_ => _ < threshold) : new Predicate<double>(_ => _ > threshold);
            var delta = inverted ? new Func<int, int>(_ => 0) : new Func<int, int>(_ => vStats.Length - _ > 30 ? 30 : vStats.Length - _);

            bool inColumn = false;
            bool first = true;

            for (int i = 0; i < vStats.Length; i++)
            {
                if (predt((double)vStats[i] / vMax))
                {
                    step = 0;

                    if (!inColumn)
                    {
                        inColumn = true;
                        result.Add(new CropLine(i));
                    }

                    if (first)
                        first = false;
                }
                else if (inColumn)
                {
                    var d = delta(i);
                    if (++step >= d)
                    {
                        inColumn = false;
                        result.Add(new CropLine(i - d));
                    }
                }
            }

            return result;
        }

        public List<CropLine> CalculateHorizontalBorders(double threshold, double rowThreshold, int rowHeight)
        {
            var result = new List<CropLine>();

            int step = 0;
            bool inColumn = false;
            bool first = true;

            for (int i = 0; i < hStats.Length; i++)
            {
                var k = (double)hStats[i] / hMax;

                if ((i != 0 && i != hStats.Length - 1) && ((double)hlStats[i].data / vStats.Length > rowThreshold))
                {
                    result.Add(new CropLine(i, true));
                    continue;
                }

                if (k > threshold)
                {
                    step = 0;

                    if (!inColumn)
                    {
                        inColumn = true;
                        result.Add(new CropLine(i, false));
                    }

                    if (first)
                        first = false;
                }
                else
                {
                    if (inColumn)
                    {
                        var delta = (hStats.Length - i > rowHeight) ? rowHeight : hStats.Length - i;
                        if (++step >= delta)
                        {
                            inColumn = false;
                            result.Add(new CropLine(i - delta, false));
                        }
                    }
                }
            }

            return result;
        }

        public static List<CropLine> NormalizeVerticalBorders(List<CropLine> raw, Size imgSize, int delta = 30, int minW = 120)
        {
            var buffer = new List<CropLine>();

            for (int i = 0; i < raw.Count; i += 2)
            {
                var l = raw[i + 1].coord - raw[i].coord;

                if (l > minW)
                {
                    buffer.Add(raw[i]);
                    buffer.Add(raw[i + 1]);
                }
            }

            var normalized = new List<CropLine>();

            normalized.Add(new CropLine(buffer[0].coord > delta ? buffer[0].coord - delta : 0));

            for (int i = 2; i < buffer.Count; i += 2)
            {
                // i and i-1
                var l = buffer[i].coord - buffer[i - 1].coord;

                if (l > 60)
                {
                    normalized.Add(new CropLine(buffer[i].coord - 30));
                    normalized.Add(new CropLine(buffer[i - 1].coord + 30));
                }
                else
                {
                    normalized.Add(new CropLine(buffer[i].coord - l / 2));
                    normalized.Add(new CropLine(buffer[i - 1].coord + l / 2));
                }
            }
            normalized.Add(new CropLine(imgSize.Width - buffer.Last().coord < delta ? imgSize.Width : buffer.Last().coord + delta));

            return normalized;
        }

        public static List<CropLine> NormalizeHorizontalBorders(List<CropLine> raw, Size imgSize, int delta = 15, int hdelta = 0, int maxH = 50)
        {
            var buffer = new List<CropLine>(raw);

            var index = 0;

            while (index < buffer.Count - 1)
            {
                if (buffer[index].hard && !buffer[index + 1].hard)
                {
                    if (buffer[index + 1].coord - buffer[index].coord < maxH)
                    {
                        buffer.RemoveAt(index + 1);
                        continue;
                    }
                }

                index++;
            }

            var result = new CropLine[2];
            maxH = -1;

            for (int i = 1; i < buffer.Count; i++)
            {
                // i and i-1, again

                var l = buffer[i].coord - buffer[i - 1].coord;
                if (l > maxH)
                {
                    maxH = l;
                    result[0] = buffer[i - 1];
                    result[1] = buffer[i];
                }
            }

            var wdelta = result[0].hard ? hdelta : delta;
            result[0].coord = (result[0].coord > wdelta) ? result[0].coord - wdelta : 0;

            wdelta = result[1].hard ? hdelta : delta;
            result[1].coord = (imgSize.Height - result[1].coord > wdelta) ? result[1].coord + wdelta : imgSize.Height;

            return new List<CropLine>(result);
        }

        public List<IndexValuePair> CalculateVerticalExtremums(Size imgSize, double threshold = 200, int leftBorder = 30, int rightBorder = 20)
        {
            // extremums

            var result = new int[vStats.Length];

            result[0] = -1;

            for (int i = 1; i < vStats.Length; i++)
                result[i] = vStats[i] - vStats[i - 1];

            var min = result.Min();

            for (int i = 1; i < vStats.Length; i++)
                result[i] = result[i] - min;

            // average

            var average = result.Average();

            int[] buffer = (int[])result.Clone();

            for (int i = 1; i < result.Length; i++)
                if (Math.Abs(result[i] - average) <= threshold)
                    result[i] = -1;

            var lastI = 0;
            for (int i = 1; i < result.Length; i++)
                if (result[i] != -1)
                {
                    if (i - lastI < 30)
                        for (int j = lastI + 1; j < i; j++)
                            result[j] = buffer[j];

                    lastI = i;
                }

            return result.Select((x, i) => new IndexValuePair(i, x)).Where(x => x.value > 0 && (x.index > leftBorder && x.index < imgSize.Width - rightBorder)).ToList();
        }

        public static List<IndexValuePair> NormalizeExtremums(List<IndexValuePair> extrems, Size imgSize)
        {
            var nExts = new List<int>();
            var buffer = new List<IndexValuePair>();

            //nExts.Add(0);
            nExts.Add(extrems[0].index);
            buffer.Add(extrems[0]);

            for (int i = 1; i < extrems.Count; i++)
            {
                var last = nExts.Last();
                var l = extrems[i].index - last;

                if (l > 120)
                {
                    if (buffer.Any())
                    {
                        nExts.RemoveAt(nExts.Count - 1);
                        nExts.Add(Averrage(buffer));
                        buffer.Clear();
                    }

                    nExts.Add(extrems[i].index);
                }
                else
                {
                    buffer.Add(extrems[i]);
                }
            }
            if (buffer.Any())
            {
                nExts.RemoveAt(nExts.Count - 1);
                nExts.Add(Averrage(buffer));
                buffer.Clear();
            }
            //nExts.Add(imgSize.Width);

            var index = 0;
            while (index < nExts.Count - 1)
                if (nExts[index + 1] - nExts[index] < 200)
                    nExts.RemoveAt(index + 1);
                else
                    index++;

            extrems = new List<IndexValuePair>(nExts.Select((x, i) => new IndexValuePair(i, x)));
            return extrems;
        }

        private static int Averrage(List<IndexValuePair> buffer)
        {
            return (int)Math.Round(buffer.Select(x => x.index).Average());
        }

        private static int WeightedAverrage(List<IndexValuePair> buffer)
        {
            var max = buffer.Max(x => x.value);
            var bufferIn = buffer.Select(x => new IndexValuePairF(x.index, (float)x.value / max)).ToList();
            var av = bufferIn.Sum(x => x.value * x.index) / bufferIn.Sum(x => x.value);
            return (int)Math.Round(av);
        }
    }

    internal struct LHist
    {
        internal int data;
        internal List<Border> borders;

        public override string ToString()
        {
            return $"{data} [{string.Join(", ", borders)}]";
        }
    }

    internal class Border
    {
        internal int start;
        internal int end;

        public override string ToString()
        {
            return $"{start} - {end}";
        }
    }
}
