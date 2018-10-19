using Emgu.CV;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RecognitionCore
{
    public class Logger
    {
        private string _session = string.Empty;
        private string _subFolder = string.Empty;
        private string _logsDir = "logs";

        #region Singletone

        private static Logger _instance = null;

        public static Logger Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new Logger();
                }

                return _instance;
            }
        }

        #endregion

        public void SaveStep(IImage image, string title)
        {
            if (_session == string.Empty)
                StartNewSession();

            if (!title.EndsWith(".png"))
                title += ".png";

            if (!Directory.Exists(Path.Combine(_logsDir, _session, _subFolder)))
                Directory.CreateDirectory(Path.Combine(_logsDir, _session, _subFolder));

            image.Save(Path.Combine(_logsDir, _session, _subFolder, title));
        }

        public void SetSubFolder(string sub)
        {
            _subFolder = sub;
        }

        public void StartNewSession()
        {
            _session = DateTime.Now.ToString("MM-dd-yyyy HH-mm-ss");

            if (!Directory.Exists("logs"))
                Directory.CreateDirectory("logs");
        }
    }
}
