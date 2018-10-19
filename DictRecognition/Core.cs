using Emgu.CV;
using Emgu.CV.Structure;
using RecognitionCore.Data;
using RecognitionCore.Data.Enums;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace RecognitionCore
{
    public static class Core
    {
        private static Dictionary<LibraryType, string[]> _divideSigns = new Dictionary<LibraryType, string[]>()
        {
            {
                LibraryType.Kamus, new string[]
                {
                    "Patterns/divide_template.png",
                    "Patterns/divide_template_2.png",
                    "Patterns/divide_template_3.png",
                    "Patterns/divide_template_4.png",
                    "Patterns/divide_template_5.png",
                    "Patterns/divide_template_6.png",
                    "Patterns/divide_template_7.png",
                    "Patterns/divide_template_8.png",
                    "Patterns/divide_template_9.png",
                }
            },
            {
                LibraryType.Lexicon, new string[]
                {
                    "Patterns/lx_divide_template.png",
                    "Patterns/lx_divide_template_2.png",
                    "Patterns/lx_divide_template_3.png",
                    "Patterns/lx_divide_template_4.png",
                    "Patterns/lx_divide_template_5.png",
                }
            },
        };

        public static EntriesGrid Process(string pathToFile, LibraryType library)
        {
            var result = new EntriesGrid();

            // Load images
            using (var imgPageSource = new Image<Bgr, byte>(pathToFile))
            using (var imgPageGray = imgPageSource.Convert<Gray, byte>())
            {
                var pageSize = imgPageSource.Size;

                Logger.Instance.SaveStep(imgPageSource, "00_page_source");
                Logger.Instance.SaveStep(imgPageGray, "00_page_gray");

                // Image denoising
                ProcessImage.Denoise(imgPageGray);
                Logger.Instance.SaveStep(imgPageGray, "00_page_gray_denoised");

                List<CropLine> vBorders;
                List<CropLine> hBorder;
                // Thresholding
                using (var imgPageThresholded = ProcessImage.TresholdingAdative(imgPageGray, 13))
                {
                    Logger.Instance.SaveStep(imgPageGray, "00_page_gray_thresholded");

                    // Calculate histogram
                    using (var imgSourceHist = ImageHelper.ExtendImage(imgPageThresholded, 240, 240))
                    {
                        var sourceHist = imgPageThresholded.CalculateHist();
                        sourceHist.NegativeZeroTransform(horizontal: true);

                        imgSourceHist.VisualizeHist(sourceHist, horizontal: true, vertical: true);
                        Logger.Instance.SaveStep(imgSourceHist, "01_page_histogram");

                        // Looking for a border of the interested area
                        vBorders = sourceHist.CalculateVerticalBorders(.1d);
                        hBorder = sourceHist.CalculateHorizontalBorders(.05d, .5d, 20);

                        using (var imgAreaBorders = imgSourceHist.Copy())
                        {
                            imgAreaBorders.VisualizeLines(vBorders.Select(x => x.coord).ToArray(), hBorder.Select(x => x.coord).ToArray());
                            Logger.Instance.SaveStep(imgAreaBorders, "01_page_interested_area_borders");
                        }

                        // Borders filtration
                        vBorders = Histogram.NormalizeVerticalBorders(vBorders, pageSize);
                        hBorder = Histogram.NormalizeHorizontalBorders(hBorder, pageSize);

                        using (var imgAreaBorders = imgSourceHist.Copy())
                        {
                            imgAreaBorders.VisualizeLines(vBorders.Select(x => x.coord).ToArray(), hBorder.Select(x => x.coord).ToArray(), color: new MCvScalar(255, 0, 0));
                            Logger.Instance.SaveStep(imgAreaBorders, "01_page_interested_area_borders_normalized");
                        }
                    }
                }

                RotationData rotationData;
                // Get images for an interested area
                using (var imgInterestedSource = ImageHelper.CutImage(imgPageSource, vBorders.Take(2).ToArray(), hBorder.Take(2).ToArray()))
                using (var imgInterestedGray = ImageHelper.CutImage(imgPageGray, vBorders.Take(2).ToArray(), hBorder.Take(2).ToArray()))
                {
                    var interestedSize = imgInterestedSource.Size;
                    Logger.Instance.SaveStep(imgInterestedGray, "02_page_interested_area");

                    // Thresholding
                    using (var imgInterestedThresholded = ProcessImage.TresholdingWithFilter(imgInterestedGray))
                    {
                        Logger.Instance.SaveStep(imgInterestedThresholded, "02_page_interested_area_thresholded");

                        // Close image (erode + delitade)
                        using (var closedImage = ProcessImage.CloseImage(imgInterestedThresholded))
                        {
                            Logger.Instance.SaveStep(closedImage, "02_page_interested_area_thresholded_closed");

                            // old - 200
                            LineSegment2D[] lines = closedImage.HoughLinesBinary(/*rhoRes*/ 1, /*thetaRes*/ Math.PI / 180, /*threshold*/ 200, /*interestedSize.Height / 12d*/ 300, /*gap*/10)[0];
                            // Looking for a rotetion angle to correct an image
                            rotationData = ProcessImage.CalculateAngle(closedImage, lines);

                            using (var imgRotationLines = imgInterestedThresholded.CopyBlank())
                            {
                                imgRotationLines.VisualizeLines(lines);
                                Logger.Instance.SaveStep(imgRotationLines, "02_page_rotation_lines");
                            }
                        }
                    }
                }

                // Rotate all images
                using (var imgInterestedSource = ImageHelper.CutImage(imgPageSource.Rotate(rotationData.angle, new Bgr(255, 255, 255)), vBorders.Take(2).ToArray(), hBorder.Take(2).ToArray()))
                using (var imgInterestedGray = ImageHelper.CutImage(imgPageGray.Rotate(rotationData.angle, new Gray(255)), vBorders.Take(2).ToArray(), hBorder.Take(2).ToArray()))
                using (var imgInterestedThresholded = ProcessImage.TresholdingAdative(imgInterestedGray, 7))
                using (var imgInterestedThresholdedClosed = ProcessImage.CloseImage(imgInterestedThresholded, erode: 1))
                {
                    var rotatedHist = imgInterestedThresholdedClosed.CalculateHist();
                    var rotatedSize = imgInterestedSource.Size;

                    Logger.Instance.SaveStep(imgInterestedThresholdedClosed, $"02_page_rotated_page_closed");

                    using (var imgInterestedHist = ImageHelper.ExtendImage(imgInterestedThresholded, 240, 240))
                    {
                        imgInterestedHist.VisualizeHist(rotatedHist, horizontal: true, vertical: true);
                        Logger.Instance.SaveStep(imgInterestedHist, "02_page_rotated_page_hist");

                        // Looking for a border of the interested area
                        vBorders = rotatedHist.CalculateVerticalBorders(.1d);
                        using (var imgAreaBorders = imgInterestedSource.Copy())
                        {
                            imgAreaBorders.VisualizeLines(vBorders.Select(x => x.coord).ToArray(), null, 0, 0);
                            Logger.Instance.SaveStep(imgAreaBorders, "03_page_interested_area_borders");
                        }

                        vBorders = Histogram.NormalizeVerticalBorders(vBorders, rotatedSize);
                        using (var imgAreaBorders = imgInterestedSource.Copy())
                        {
                            imgAreaBorders.VisualizeLines(vBorders.Select(x => x.coord).ToArray(), null, 0, 0);
                            Logger.Instance.SaveStep(imgAreaBorders, "03_page_interested_area_borders_normalized");
                        }

                        foreach (var v in vBorders)
                            result.VerticalLines.Add(new VLine(v.coord, rotatedSize));
                    }

                    // Entries recognition
                    int current = 0;
                    while (current < vBorders.Count - 1)
                    {
                        using (var imgColumnSource = ImageHelper.CutImage(imgInterestedSource, new CropLine[] { vBorders[current], vBorders[current + 1] }, new CropLine[] { new CropLine(0), new CropLine(rotatedSize.Height) }))
                        using (var imgColumnGray = ImageHelper.CutImage(imgInterestedGray, new CropLine[] { vBorders[current], vBorders[current + 1] }, new CropLine[] { new CropLine(0), new CropLine(rotatedSize.Height) }))
                        using (var imgColumnThresholded = ImageHelper.CutImage(imgInterestedThresholded, new CropLine[] { vBorders[current], vBorders[current + 1] }, new CropLine[] { new CropLine(0), new CropLine(rotatedSize.Height) }))
                        using (var imgColumnThresholdedClosed = ImageHelper.CutImage(imgInterestedThresholdedClosed, new CropLine[] { vBorders[current], vBorders[current + 1] }, new CropLine[] { new CropLine(0), new CropLine(rotatedSize.Height) }))
                        {
                            var columnSize = imgColumnSource.Size;

                            // Calculate column histogram
                            var columnHist = imgColumnThresholdedClosed.CalculateHist();
                            columnHist.NegativeZeroTransform(horizontal: true);
                            using (var imgColumnHist = ImageHelper.ExtendImage(imgColumnThresholded, 240, 240))
                            {
                                imgColumnHist.VisualizeHist(columnHist, horizontal: true, vertical: true);
                                Logger.Instance.SaveStep(imgColumnHist, $"04_column_{current}_histogram");
                            }

                            // Calculate extremums to detect a skiped columns
                            var extremums = columnHist.CalculateVerticalExtremums(columnSize);

                            if (extremums.Any())
                            {
                                using (var imgColumnExt = ImageHelper.ExtendImage(imgColumnThresholded, 240, 240))
                                {
                                    imgColumnExt.VisualizeExtremums(extremums, vertical: true);
                                    Logger.Instance.SaveStep(imgColumnExt, $"04_column_{current}_extrems");
                                }

                                // Normalize extremums
                                extremums.AddRange(vBorders.Select(x => new IndexValuePair(x.coord, 0)));
                                extremums = new List<IndexValuePair>(extremums.OrderBy(x => x.index));

                                var normalized = Histogram.NormalizeExtremums(extremums, columnSize);
                                using (var imgColumnExtNormalized = ImageHelper.ExtendImage(imgColumnThresholded, 240, 240))
                                {
                                    imgColumnExtNormalized.VisualizeLines(normalized.Select(x => x.value).ToArray(), null);
                                    Logger.Instance.SaveStep(imgColumnExtNormalized, $"04_column_{current}_extrems_normalized");
                                }

                                if (normalized.Count > vBorders.Count)
                                {
                                    // need to split current column into 2
                                    vBorders = new List<CropLine>(normalized.Select(_ => new CropLine(_.value)));
                                    continue;
                                }
                            }

                            hBorder = columnHist.CalculateHorizontalBorders(.05d, .85d, 30);
                            using (var imgColumnHorizontalBorders = ImageHelper.ExtendImage(imgColumnThresholded, 240, 240))
                            {
                                imgColumnHorizontalBorders.VisualizeLines(null, hBorder.Select(_ => _.coord).ToArray());
                                Logger.Instance.SaveStep(imgColumnHorizontalBorders, $"04_column_{current}_horizontal_borders");
                            }

                            hBorder = Histogram.NormalizeHorizontalBorders(hBorder, columnSize);
                            using (var imgColumnHorizontalBordersNormalized = ImageHelper.ExtendImage(imgColumnThresholded, 240, 240))
                            {
                                imgColumnHorizontalBordersNormalized.VisualizeLines(null, hBorder.Select(_ => _.coord).ToArray());
                                Logger.Instance.SaveStep(imgColumnHorizontalBordersNormalized, $"04_column_{current}_horizontal_borders_normalized");
                            }

                            //hBorder = new List<CropLine>();

                            using (var imgColumnFinalSource = ImageHelper.CutImage(imgColumnSource, new CropLine[] { new CropLine(0), new CropLine(columnSize.Width) }, new CropLine[] { hBorder[0], hBorder[1] }))
                            using (var imgColumnFinalGray = ImageHelper.CutImage(imgColumnGray, new CropLine[] { new CropLine(0), new CropLine(columnSize.Width) }, new CropLine[] { hBorder[0], hBorder[1] }))
                            {
                                // Save the column image
                                Logger.Instance.SaveStep(imgColumnFinalSource, $"05_column_{current}_final_source");
                                columnSize = imgColumnFinalSource.Size;

                                List<int> hMap = new List<int>();

                                switch (library)
                                {
                                    #region Kamus
                                    case LibraryType.Kamus:
                                        using (var imgColumnFinalBlured = imgColumnFinalGray.SmoothBlur(11, 11))
                                        using (var imgColumnFinalBluredThresholded = imgColumnFinalBlured.ThresholdBinaryInv(new Gray(190), new Gray(255)))
                                        {
                                            Logger.Instance.SaveStep(imgColumnFinalBlured, $"05_column_{current}_blured");
                                            Logger.Instance.SaveStep(imgColumnFinalBluredThresholded, $"05_column_{current}_blured_thresholded");
                                            //columnHist = imgColumnFinalThresholded.CalculateHist();

                                            var match = imgColumnFinalGray.LookingForPatterns(_divideSigns[LibraryType.Kamus], .8, columnSize.Width * 6 / 7);

                                            using (var imgPatterns = imgColumnFinalGray.Convert<Bgr, byte>())
                                            {
                                                imgPatterns.VisualizeSegments(match.ToArray(), dh: 0, dv: 0);
                                                Logger.Instance.SaveStep(imgPatterns, $"05_column_{current}_matching_templates");
                                            }

                                            var lines = ProcessImage.CalculateLines(imgColumnFinalBluredThresholded, percentThreshold: 5);

                                            // Calculate central and averaged lines
                                            var dLinesCentral = new List<int>();
                                            var dLinesCentralS = new List<int>();
                                            hMap = match.Select(x => x.start.Y).ToList();
                                            foreach (var list in lines)
                                            {
                                                dLinesCentral.Add((int)list.Average());
                                                dLinesCentralS.Add((list.Sum() / list.Count));
                                            }

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeLines(null, lines.SelectMany(x => x).ToArray(), 0, 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_all_lines");
                                            }

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeLines(null, dLinesCentral.ToArray(), 0, 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_horizontal_borders");
                                            }

                                            ProcessImage.NormalizeLines(hMap, dLinesCentral, dLinesCentralS);

                                            hMap.AddRange(new int[] { dLinesCentralS.First(), dLinesCentralS.Last() });
                                        }
                                        break;
                                    #endregion
                                    #region Lexicon
                                    case LibraryType.Lexicon:
                                        using (var imgColumnFinalBlured = imgColumnFinalGray.SmoothBlur(11, 11))
                                        using (var imgColumnFinalBluredThresholded = imgColumnFinalBlured.ThresholdBinaryInv(new Gray(230), new Gray(255)))
                                        using (var imgColumnFinalBluredThresholdedClosed = imgColumnFinalBluredThresholded.Erode(2))
                                        {
                                            columnHist = imgColumnFinalBluredThresholded.CalculateHist();
                                            columnSize = imgColumnFinalBluredThresholded.Size;

                                            Logger.Instance.SaveStep(imgColumnFinalBlured, $"05_column_{current}_0_blured");
                                            Logger.Instance.SaveStep(imgColumnFinalBluredThresholded, $"05_column_{current}_0_blured_thresholded");
                                            Logger.Instance.SaveStep(imgColumnFinalBluredThresholdedClosed, $"05_column_{current}_0_blured_thresholded_closed");

                                            var match = imgColumnFinalGray.LookingForPatterns(_divideSigns[LibraryType.Lexicon], .8);

                                            using (var imgPatterns = imgColumnFinalGray.Convert<Bgr, byte>())
                                            {
                                                imgPatterns.VisualizeSegments(match.ToArray(), dh: 0, dv: 0);
                                                Logger.Instance.SaveStep(imgPatterns, $"05_column_{current}_1_matching_templates");
                                            }

                                            var sPoints = ProcessImage.LookingForAStartPoints(imgColumnFinalBluredThresholded);
                                            var ePoints = ProcessImage.LookingForAnEndPoints(imgColumnFinalBluredThresholded);

                                            var lines = ProcessImage.CalculateLines(imgColumnFinalBluredThresholdedClosed, threshold: 200, percentThreshold: 15, ePoints: ePoints);

                                            var dLinesCentral = new List<int>();
                                            foreach (var list in lines)
                                                dLinesCentral.Add((int)list.Average());

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeLines(null, lines.SelectMany(x => x).ToArray(), 0, 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_2_all_lines");
                                            }

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeLines(null, dLinesCentral.ToArray(), 0, 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_2_horizontal_borders");
                                            }

                                            // Calculate entries segments
                                            var segments = new List<PageSegment>();

                                            segments.Add(new PageSegment(new Point(0, 0), new Point(columnSize.Width, dLinesCentral[0]), sPoints));
                                            for (int iter = 1; iter < dLinesCentral.Count; iter++)
                                            {
                                                segments.Add(new PageSegment(segments.Last().end, new Point(columnSize.Width, dLinesCentral[iter]), sPoints));
                                            }
                                            segments.Add(new PageSegment(new Point(0, dLinesCentral.Last()), new Point(columnSize.Width, columnSize.Height), sPoints));

                                            var maxSP = segments.Where(x => x.startPoint != 0).Select(x => x.startPoint).Max();
                                            var minSP = segments.Where(x => x.startPoint != 0).Select(x => x.startPoint).Min();
                                            var averageStartPoint = (maxSP + minSP) / 2;

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeStartPoints(sPoints, average: averageStartPoint, dh: 0, dv: 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_2_start_points");
                                            }

                                            using (var imgShow = imgColumnFinalSource.Copy())
                                            {
                                                imgShow.VisualizeStartPoints(ePoints, dh: 0, dv: 0);
                                                Logger.Instance.SaveStep(imgShow, $"05_column_{current}_2_end_points");
                                            }

                                            // Looking for an entries
                                            foreach (var segm in segments)
                                                if (segm.startPoint >= averageStartPoint)
                                                    hMap.Add(segm.start.Y);

                                            foreach (var reg in match)
                                            {
                                                hMap.RemoveAll(x => x > reg.start.Y && x < reg.end.Y - 5);
                                                hMap.Add(reg.start.Y);
                                            }

                                            hMap.AddRange(new int[] { dLinesCentral.First(), dLinesCentral.Last() });
                                        }
                                        break;
                                        #endregion
                                }

                                // Cut entries
                                hMap = hMap.Distinct().ToList();
                                hMap.Sort();

                                using (var imgFinalEntries = imgColumnFinalSource.Copy())
                                {
                                    imgFinalEntries.VisualizeLines(null, hMap.ToArray(), 0, 0);
                                    Logger.Instance.SaveStep(imgFinalEntries, $"06_column_{current}_final_entries");
                                }

                                int entryN = 0;
                                for (int i = 0; i < hMap.Count - 1; i++)
                                {
                                    // Skip too small entries
                                    if (hMap[i + 1] - hMap[i] < 20)
                                        continue;

                                    using (var imgEntry = ImageHelper.CutImage(imgColumnFinalSource, new CropLine[] { new CropLine(0), new CropLine(columnSize.Width) }, new CropLine[] { new CropLine(hMap[i]), new CropLine(hMap[i + 1]) }))
                                        Logger.Instance.SaveStep(imgEntry, $"07_column_{current}_entry_{++entryN}");
                                }
                            }

                            current++;
                        }
                    }
                }
            }

            return result;
        }
    }
}
