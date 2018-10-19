using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore.Data
{
    public struct SpaceSegment
    {
        internal List<IndexValuePairF> indexes;
        internal float sumpercent;
        internal float averrage;

        public SpaceSegment(List<IndexValuePairF> indexes, float sumpercent)
        {
            this.indexes = indexes;
            this.sumpercent = sumpercent;
            this.averrage = sumpercent / indexes.Count();
        }

        public override string ToString()
        {
            return $"[{indexes.Count}] - {sumpercent} / {averrage}";
        }
    }
}
