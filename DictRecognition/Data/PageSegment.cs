using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore.Data
{
    public struct PageSegment
    {
        internal System.Drawing.Point start;
        internal System.Drawing.Point end;
        internal int startPoint;
        internal Size size;

        public PageSegment(System.Drawing.Point start, System.Drawing.Point end, int[] starts = null)
        {
            this.start = start;
            this.end = end;
            this.size = new Size(end.X - start.X, end.Y - start.Y);

            if (starts != null)
            {
                var segment = starts.Skip(this.start.Y).Take(this.size.Height).Where(x => x != 0);

                var delta = (int)Math.Ceiling((double)segment.Count() / 3);
                var central = segment.Skip(delta).Take(delta);

                this.startPoint = central.Any() ? (int)central.Min() : 0;
                if (this.startPoint > 100)
                    this.startPoint = 0;
            }
            else
                this.startPoint = 0;
        }

        public Rectangle ToRect()
        {
            return new Rectangle(start, size);
        }

        public Rectangle ToRect(int dx, int dy)
        {
            return new Rectangle(new Point(start.X + dx, start.Y + dy), size);
        }

        public override string ToString()
        {
            return $"[{start} - {end}]. SP:{startPoint}";
        }
    }
}
