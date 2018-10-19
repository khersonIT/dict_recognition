using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using RecognitionCore.Data;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore
{
    internal static class ProcessImage
    {
        public static void Denoise(IImage img)
        {
            CvInvoke.FastNlMeansDenoising(img, img, 12);
        }

        public static Image<Gray, byte> TresholdingAdative(Image<Gray, byte> image, int blockSize = 11)
        {
            var tres = image.CopyBlank();
            CvInvoke.AdaptiveThreshold(image, tres, 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary, blockSize, -1);
            return tres;
        }

        public static Image<Gray, byte> TresholdingWithFilter(Image<Gray, byte> image, int size = 20, double c = .3d)
        {
            var smoothedGray = image.SmoothBlur(size, size);

            var bw = smoothedGray - image - c;
            return bw.ThresholdBinary(new Gray(0), new Gray(255));
        }

        public static Image<Gray, byte> CloseImage(Image<Gray, byte> img, int size = 2, int erode = 0, int dilate = 0)
        {
            var el = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(2 * size + 1, 2 * size + 1), new Point(size, size));

            var res = img.MorphologyEx(MorphOp.Close, el, new Point(size, size), 1, BorderType.Default, new MCvScalar(0));

            if (erode != 0)
                res = res.Erode(erode);
            if (dilate != 0)
                res = res.Dilate(dilate);

            return res;
        }

        public static RotationData CalculateAngle(Image<Gray, byte> img, LineSegment2D[] lines, int theta = 3, int deltaL = 10)
        {
            int count = 0;
            var angle = 0d;
            var langle = new List<double>();

            var lineGroups = new Dictionary<int, List<Point>>();

            for (int i = 0; i < lines.Length; i++)
            {
                var a = Math.Atan2(lines[i].P2.X - lines[i].P1.X, lines[i].P2.Y - lines[i].P1.Y) * 180 / Math.PI;

                if (a > 90)
                    a = -(180 - a);

                if (a > -theta && a < theta)
                {
                    var color = a > 0 ? new Bgr(255, 0, 0) : new Bgr(0, 0, 255);

                    var dx = (lines[i].P1.X + lines[i].P2.X) / 2;

                    var currentGroup = lineGroups.FirstOrDefault(x => Math.Abs(x.Key - dx) <= deltaL);
                    if (currentGroup.Key == 0 && !lineGroups.ContainsKey(dx))
                    {
                        lineGroups.Add(dx, new List<Point>()
                            {
                                lines[i].P1,
                                lines[i].P2
                            });
                    }
                    else
                    {
                        currentGroup.Value.Add(lines[i].P1);
                        currentGroup.Value.Add(lines[i].P2);
                    }

                    angle += a;
                    count++;
                }

                langle.Add(a);
            }

            var endLinesL = new List<LineSegment2D>();
            int lineCount = 0;
            angle = 0;
            foreach (var points in lineGroups.Values)
            {
                var minX = points.Min(x => x.X);
                var minY = points.Min(x => x.Y);

                var maxX = points.Max(x => x.X);
                var maxY = points.Max(x => x.Y);

                var topPoint = points.FirstOrDefault(p => p.Y == points.Min(x => x.Y));
                var bottomPoint = points.FirstOrDefault(p => p.Y == points.Max(x => x.Y));

                LineSegment2D line;

                if (bottomPoint.X < topPoint.X)
                    line = new LineSegment2D(new Point(maxX, minY), new Point(minX, maxY));
                else
                    line = new LineSegment2D(new Point(minX, minY), new Point(maxX, maxY));

                endLinesL.Add(line);
            }

            var endLines = endLinesL.ToArray();

            foreach (var line in endLines)
            {
                if (line.Length < endLines.Max(x => x.Length) * 3 / 4d)
                    continue;

                var a = Math.Atan2(line.P2.X - line.P1.X, line.P2.Y - line.P1.Y);
                a = a * 180 / Math.PI;

                lineCount++;
                angle += a;
            }

            return new RotationData((lineGroups.Count > 0) ? angle / lineCount : 0, endLines);
        }

        public static int[] LookingForAStartPoints(Image<Gray, byte> img, bool horizontal = true)
        {
            var startD = 12;

            int[] result;
            if (horizontal)
                result = new int[img.Height];
            else
                result = new int[img.Width];

            var data = img.Data;

            for (int r = 0; r < img.Rows; r++)
                for (int c = startD; c < img.Cols / 5; c++)
                    if (data[r, c, 0] == 255)
                    {
                        result[r] = c;
                        break;
                    }

            return result;
        }

        public static int[] LookingForAnEndPoints(Image<Gray, byte> img, bool horizontal = true)
        {
            var endD = 12;

            int[] result;
            if (horizontal)
                result = new int[img.Height];
            else
                result = new int[img.Width];

            var data = img.Data;

            for (int r = 0; r < img.Rows; r++)
                for (int c = img.Width - endD; c >= 0; c--)
                    if (data[r, c, 0] == 255)
                    {
                        result[r] = c;
                        break;
                    }

            return result;
        }

        public static List<List<int>> CalculateLines(Image<Gray, byte> img, int threshold = 160, int percentThreshold = 5, int[] ePoints = null, int dh = 15)
        {
            var res = new List<SpaceSegment>();

            var horProj = new Matrix<int>(img.Rows, 1);
            CvInvoke.Reduce(img, horProj, ReduceDimension.SingleCol, ReduceType.ReduceSum, horProj.Mat.Depth);

            var all = new List<IndexValuePairF>();

            int iter = 0;
            var max = horProj.Data.Cast<int>().Max();
            double k = max;

            var maxI = horProj.Data.Cast<int>().ToList().IndexOf(max);

            if (ePoints != null)
            {
                k = (double)max / ePoints[maxI];
                var replMax = k * ePoints[maxI];
            }
            foreach (var data in horProj.Data)
            {
                var percent = 100 * (double)data / (k * ((ePoints == null) ? 1 : ePoints[iter]));

                if (percent < percentThreshold)
                    all.Add(new IndexValuePairF(iter, percent));

                iter++;
            }

            var allspercents = new List<double>();

            var buffer = new List<int>();
            var sumpercent = 0f;
            var last = new IndexValuePairF(0, 0);

            foreach (var linen in all)
            {
                if (!buffer.Any())
                {
                    last = linen;
                    buffer.Add(linen.index);
                    sumpercent += linen.value;
                    continue;
                }

                if (linen.index - last.index > dh)
                {
                    res.Add(new SpaceSegment(new List<IndexValuePairF>(buffer.Select(x => new IndexValuePairF(x, 100 * (double)horProj.Data[x, 0] / max))), sumpercent));
                    buffer.Clear();

                    allspercents.Add(sumpercent);
                    sumpercent = 0;
                }

                last = linen;
                buffer.Add(linen.index);
                sumpercent += linen.value;
            }

            if (buffer.Any())
            {
                res.Add(new SpaceSegment(new List<IndexValuePairF>(buffer.Select(x => new IndexValuePairF(x, 100 * (double)horProj.Data[x, 0] / max))), sumpercent));
                allspercents.Add(sumpercent);
            }

            return res.Select(x => x.indexes.Select(y => y.index).ToList()).ToList();
        }

        public static void NormalizeLines(List<int> hMap, List<int> cLines, List<int> cLinesS)
        {
            for (int h = 0; h < hMap.Count; h++)
            {
                for (int c = 0; c < cLines.Count; c++)
                {
                    if (c + 1 == cLinesS.Count)
                    {
                        //CvInvoke.Line(temp, new Point(0, hPoints[h]), new Point(width, hPoints[h]), new MCvScalar(0, 0, 255), 2);
                        //CvInvoke.Line(temp, new Point(0, dLinesCentral[c]), new Point(width, dLinesCentral[c]), new MCvScalar(255, 0, 0), 2);
                        hMap[h] = cLines[c];
                        break;
                    }
                    else if (hMap[h] > cLines[c] && hMap[h] < cLines[c + 1])
                    {
                        //CvInvoke.Line(temp, new Point(0, hPoints[h]), new Point(width, hPoints[h]), new MCvScalar(0, 0, 255), 2);
                        var newH = (cLines[c + 1] - hMap[h] >= hMap[h] - cLines[c]) ? cLines[c] : cLines[c + 1];

                        hMap[h] = (newH > 20) ? hMap[h] - 10 : newH;

                        //CvInvoke.Line(temp, new Point(0, hMap[h]), new Point(width, hMap[h]), new MCvScalar(255, 0, 255), 2);
                        break;
                    }
                }
            }
        }
    }

    internal static class ImageHelper
    {
        public static IEnumerable<PageSegment> LookingForPatterns(this Image<Gray, byte> img, string[] patterns, double matchingThreshold, double border = -1)
        {
            var resultPoints = new List<PageSegment>();
            int? last = null;

            foreach (var p in patterns)
            {
                using (var imgPattern = new Image<Gray, byte>(p))
                {
                    using (var imgLocator = img.MatchTemplate(imgPattern, TemplateMatchingType.CcoeffNormed))
                    {
                        var patternSize = imgPattern.Size;
                        float[,,] data = imgLocator.Data;

                        for (int r = 0; r < imgLocator.Rows; r++)
                            for (int c = 0; c < imgLocator.Cols; c++)
                                if (data[r, c, 0] > matchingThreshold && c > border)
                                    if (last == null || r - last > patternSize.Height)
                                    {
                                        resultPoints.Add(new PageSegment(new Point(c, r), new Point(c + patternSize.Width, r + patternSize.Height)));
                                        last = r;
                                    }
                    }
                }
                last = null;
            }

            // Sort and filter founded
            resultPoints.Sort((segm1, segm2) => segm1.start.Y.CompareTo(segm2.start.Y));

            if (resultPoints.Any())
            {
                var buffer = new List<PageSegment>() { resultPoints.First() };

                for (int i = 1; i < resultPoints.Count; i++)
                    if (resultPoints[i].start.Y > resultPoints[i - 1].end.Y)
                        buffer.Add(resultPoints[i]);

                resultPoints = new List<PageSegment>(buffer);

                if (border != -1)
                {
                    // Checking patterns groups
                    var groups = new Dictionary<PageSegment, List<PageSegment>>();
                    foreach (var pat in resultPoints)
                    {
                        if (!groups.Any(_ => DLength(_.Key, pat) > 0))
                        {
                            groups.Add(pat, new List<PageSegment>() { pat });
                        }
                        else
                        {
                            var key = groups.First(_ => DLength(_.Key, pat) > 0).Key;
                            groups[key].Add(pat);
                        }
                    }

                    var finalKey = groups.Keys.OrderByDescending(_ => _.start.X).First();

                    resultPoints = groups[finalKey];
                }
            }

            return resultPoints;
        }
        private static int DLength(PageSegment p1, PageSegment p2)
        {
            return Math.Min(p1.end.X, p2.end.X) - Math.Max(p1.start.X, p2.start.X);
        }

        public static Image<Bgr, byte> ExtendImage(Image<Gray, byte> imgSource, int dw, int dh)
        {
            var imgExtented = new Image<Bgr, byte>(new Size(dw + imgSource.Size.Width, dh + imgSource.Size.Height));

            var sROI = imgExtented.ROI;
            imgExtented.ROI = new Rectangle(dw, dh, imgSource.Size.Width, imgSource.Size.Height);
            imgSource.Convert<Bgr, byte>().CopyTo(imgExtented);
            imgExtented.ROI = sROI;

            return imgExtented;
        }

        public static Image<TColor, TDepth> CutImage<TColor, TDepth>(Image<TColor, TDepth> img, CropLine[] vertical, CropLine[] horizontal)
            where TColor : struct, IColor
            where TDepth : new()
        {
            return CutImageByRegion(img, new Rectangle(vertical[0].coord, horizontal[0].coord, vertical[1].coord - vertical[0].coord, horizontal[1].coord - horizontal[0].coord));
        }

        private static Image<TColor, TDepth> CutImageByRegion<TColor, TDepth>(Image<TColor, TDepth> img, Rectangle roi, int offsetX = 0, int offsetY = 0)
            where TColor : struct, IColor
            where TDepth : new()
        {
            var newImage = new Image<TColor, TDepth>(new Size(roi.Width + offsetX, roi.Height + offsetY));

            var sourceROI = img.ROI;
            img.ROI = roi;
            img.CopyTo(newImage);
            img.ROI = sourceROI;

            return newImage;
        }
    }
}
