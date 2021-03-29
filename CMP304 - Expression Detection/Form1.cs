using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;
using Dlib = DlibDotNet.Dlib;
using DlibDotNet.Extensions;


namespace CMP304___Expression_Detection
{
    public partial class Form1 : Form
    {
        ExpressionDetectionSystem EDS = new ExpressionDetectionSystem();

        public static PictureBox image;
        public static Label progressLabel;
        public static Panel metricPanel;
        public static Label metricText;

        public Form1()
        {
            InitializeComponent();
            CenterToScreen();

            BackColor = Color.PowderBlue;

            Panel panel = CreatePanel();
            panel.SendToBack();
            this.Controls.Add(panel);

            Label title = CreateTitle();
            title.Location = new Point((panel.Size.Width - title.Size.Width) / 2, 25);
            panel.Controls.Add(title);

            image = CreatePictureBox();
            this.Controls.Add(image);

            Button extractFeatureVectorsButton = CreateButton("Extract feature vectors", new Vector2(95, 120));
            this.Controls.Add(extractFeatureVectorsButton);

            Button testDataset = CreateButton("Test dataset against model", new Vector2(95, 190));
            this.Controls.Add(testDataset);

            Button customTestButton = CreateButton("Test custom image", new Vector2(95, 260));
            this.Controls.Add(customTestButton);

            Button trainModelButton = CreateLargeButton("Train Model", new Vector2(115, 340), new Vector2(200, 60));
            this.Controls.Add(trainModelButton);

            progressLabel = CreateLabel("", new Vector2(458, 370));
            this.Controls.Add(progressLabel);

            metricPanel = CreatePanel2(new Vector2(40, 50));
            metricPanel.BringToFront();
            this.Controls.Add(metricPanel);

            metricText = CreateMetricText("yeet", new Vector2(40, 10));
            metricPanel.Controls.Add(metricText);

            Button closeMetricsButton = CreateButton("X", new Vector2(690, 0), new Vector2(30, 30));
            metricPanel.Controls.Add(closeMetricsButton);
            metricPanel.Visible = false;

            customTestButton.Click += new System.EventHandler(TestCustomFace);
            trainModelButton.Click += new System.EventHandler(TrainModel);
            closeMetricsButton.Click += new System.EventHandler(CloseMetrics);

            extractFeatureVectorsButton.Click += new System.EventHandler(ExtractFeatureVectors);
            testDataset.Click += new System.EventHandler(TestDataset);

        }

        public void TrainModel(object sender, EventArgs e)
        {
            progressLabel.Text = "Training Model...";
            CMP304___Expression_Detection.ExpressionDetectionSystem.TrainModel();
            progressLabel.Text = "";
        }

        public static void DisplayMetrics(string metrics)
        {
            if (metricPanel.InvokeRequired)
            {
                metricPanel.Invoke(new MethodInvoker(delegate 
                {
                    metricPanel.Visible = true;
                    metricPanel.BringToFront();
                    metricText.Text = metrics;
                }));
            }
            else
            {
                metricPanel.Visible = true;
                metricPanel.BringToFront();
                metricText.Text = metrics;
            }

        }

        private void ExtractFeatureVectors(object sender, EventArgs e)
        {
            using (var folderDialogue = new FolderBrowserDialog())
            {
                folderDialogue.SelectedPath = @"C:\Users\ollie\Desktop\Abertay\Year 3 Semester 2\CMP304\CMP304 - Expression Detection\CMP304 - Expression Detection\TrainingImages\";
                if (folderDialogue.ShowDialog() == DialogResult.OK)
                {
                    string filepath = folderDialogue.SelectedPath;
                    ExpressionDetectionSystem.SelectSet(filepath, CMP304___Expression_Detection.Datatype.Training);
                }
            }
        }

        private void TestDataset(object sender, EventArgs e)
        {
            using (var folderDialogue = new FolderBrowserDialog())
            {
                folderDialogue.SelectedPath = @"C:\Users\ollie\Desktop\Abertay\Year 3 Semester 2\CMP304\CMP304 - Expression Detection\CMP304 - Expression Detection\TrainingImages\";
                if (folderDialogue.ShowDialog() == DialogResult.OK)
                {
                    string filepath = folderDialogue.SelectedPath;
                    ExpressionDetectionSystem.SelectSet(filepath, CMP304___Expression_Detection.Datatype.Testing);
                }
            }
        }


        private void TestCustomFace(object sender, EventArgs e)
        {
            progressLabel.Text = "Predicting expression...";
            image.Image = null;

            var fileDialog = new OpenFileDialog();

            if(fileDialog.ShowDialog() == DialogResult.OK)
            {
                string fileToOpen = fileDialog.FileName;

                System.IO.FileInfo File = new System.IO.FileInfo(fileDialog.FileName);

                string filepath = File.DirectoryName + "/" + File.Name;

                string predictedExpression = CMP304___Expression_Detection.ExpressionDetectionSystem.TestCustomImage(filepath);

                image.Image = Image.FromFile(filepath);
                image.Visible = true;
                progressLabel.Text = "I think they are expressing " + CMP304___Expression_Detection.ExpressionDetectionSystem.TestCustomImage(filepath) + ".";

            }
        }


        private void CloseMetrics(object sender, EventArgs e)
        {
            metricPanel.Visible = false;
            metricPanel.SendToBack();
        }

        public static void SetProgress(int currentFace, int maxFaces)
        {
            if (progressLabel.InvokeRequired)
            {
                progressLabel.Invoke(new MethodInvoker(delegate { progressLabel.Text = "Processing " + currentFace + " out of " + maxFaces; }));
            }
        }

        public static void HideImage()
        {
            if (image.InvokeRequired)
            {
                image.Invoke(new MethodInvoker(delegate { image.Image = null; image.Visible = false;}));
            }

            if (progressLabel.InvokeRequired)
            {
                progressLabel.Invoke(new MethodInvoker(delegate { progressLabel.Text = "Training data processed."; }));
            }

        }

        public static void SetImage(System.Drawing.Image img)
        {
            if (image.InvokeRequired)
            {
                image.Invoke(new MethodInvoker(delegate { image.Visible = true; }));
            }

            image.Image = img;
        }

        PictureBox CreatePictureBox()
        {
            System.Windows.Forms.PictureBox pictureBox = new System.Windows.Forms.PictureBox
            {
                Size = new Size(350, 250),
                BackColor = Color.PowderBlue,
                SizeMode = PictureBoxSizeMode.Zoom,
                BorderStyle = BorderStyle.None,
                Location = new Point((400), 120),
                //Visible = false  
        };
            return pictureBox;
        }

        Button CreateButton(string text, Vector2 location)
        {
            System.Windows.Forms.Button button = new System.Windows.Forms.Button
            {
                Size = new Size(240, 50),
                Text = text,
                Font = new Font("Source Sans Pro", 12, FontStyle.Regular),
                FlatStyle = FlatStyle.Flat,
                FlatAppearance = { BorderSize = 0 },
                ForeColor = Color.White,
                BackColor = Color.CadetBlue,
                Visible = true,
                TextAlign = ContentAlignment.MiddleCenter,
                Location = new Point((int)location.x, (int)location.y)
            };

            button.FlatAppearance.BorderColor = Color.FromArgb(255, 255, 255, 255);

            return button;

        }

        Button CreateButton(string text, Vector2 location, Vector2 size)
        {
            System.Windows.Forms.Button button = new System.Windows.Forms.Button
            {
                Size = new Size((int)size.x, (int)size.y),
                Text = text,
                Font = new Font("Source Sans Pro", 10, FontStyle.Regular),
                FlatStyle = FlatStyle.Flat,
                FlatAppearance = { BorderSize = 0 },
                ForeColor = Color.White,
                BackColor = Color.CadetBlue,
                Visible = true,
                Location = new Point((int)location.x, (int)location.y)
            };

            button.FlatAppearance.BorderColor = Color.FromArgb(255, 255, 255, 255);

            return button;
        }

        Button CreateLargeButton(string text, Vector2 location, Vector2 size)
        {
            System.Windows.Forms.Button button = new System.Windows.Forms.Button
            {
                Size = new Size((int)size.x, (int)size.y),
                Text = text,
                Font = new Font("Source Sans Pro", 16, FontStyle.Bold),
                FlatStyle = FlatStyle.Flat,
                FlatAppearance = { BorderSize = 0 },
                ForeColor = Color.White,
                BackColor = Color.CadetBlue,
                Visible = true,
                Location = new Point((int)location.x, (int)location.y)
            };

            button.FlatAppearance.BorderColor = Color.FromArgb(255, 255, 255, 255);

            return button;
        }

        Label CreateTitle()
        {
            System.Windows.Forms.Label label = new System.Windows.Forms.Label
            {
                Size = new Size(800, 50),
                Text = "Facial Expression Detection System - Ollie Hall 1700066",
                Anchor = AnchorStyles.Left | AnchorStyles.Right,
                TextAlign = ContentAlignment.MiddleCenter,
                Font = new Font("Source Sans Pro", 20, FontStyle.Regular),
                ForeColor = Color.White,
                BackColor = Color.CadetBlue
            };

            return label;
        }

        Label CreateLabel(string text, Vector2 position)
        {
            System.Windows.Forms.Label label = new System.Windows.Forms.Label
            {
                Size = new Size(235, 80),
                Text = text,
                Enabled = true,
                Font = new Font("Source Sans Pro", 12, FontStyle.Bold),
                ForeColor = Color.Black,
                Location = new Point((int)position.x, (int)position.y),
                TextAlign = ContentAlignment.MiddleCenter
            };

            return label;
        }

        Label CreateMetricText(string text, Vector2 position)
        {
            System.Windows.Forms.Label label = new System.Windows.Forms.Label
            {
                Size = new Size(650, 420),
                Text = text,
                Enabled = true,
                Font = new Font("Consolas", 9, FontStyle.Regular),
                ForeColor = Color.White,
                //BackColor = Color.Black,
                Location = new Point((int)position.x, (int)position.y),
                TextAlign = ContentAlignment.TopLeft
            };

            return label;
        }

        Panel CreatePanel()
        {
            System.Windows.Forms.Panel panel = new System.Windows.Forms.Panel
            {
                Size = new Size(300, 100),
                Dock = DockStyle.Top,
                BackColor = Color.PowderBlue,
                BorderStyle = BorderStyle.None

            };
            return panel;
        }

        Panel CreatePanel2(Vector2 position)
        {
            System.Windows.Forms.Panel panel = new System.Windows.Forms.Panel
            {
                Location = new Point((int)position.x, (int)position.y),
                Size = new Size(720, 350),
                BackColor = Color.DarkSlateGray,
                BorderStyle = BorderStyle.Fixed3D,
            };
            return panel;
        }

    }
}
