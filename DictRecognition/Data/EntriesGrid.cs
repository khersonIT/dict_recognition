using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore.Data
{
    public class EntriesGrid
    {
        // TODO : add h map here
        public List<HLine> HorizontalLines { get; set; } = new List<HLine>();
        public List<VLine> VerticalLines { get; set; } = new List<VLine>();
    }
}