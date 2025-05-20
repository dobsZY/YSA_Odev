using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YSAOdev
{
    public class YSA
    {
        int girdiSayisi, gizliSayisi, ciktiSayisi;
        double[,] agirlikGizli;
        double[,] agirlikCikti;
        double[] biasGizli;
        double[] biasCikti;
        double ogrenmeOrani;
        Random rnd = new Random();

        public YSA(int girdiSayisi, int gizliSayisi, int ciktiSayisi, double ogrenmeOrani)
        {
            this.girdiSayisi = girdiSayisi;
            this.gizliSayisi = gizliSayisi;
            this.ciktiSayisi = ciktiSayisi;
            this.ogrenmeOrani = ogrenmeOrani;

           
            agirlikGizli = new double[gizliSayisi, girdiSayisi];
            agirlikCikti = new double[ciktiSayisi, gizliSayisi];
            biasGizli = new double[gizliSayisi];
            biasCikti = new double[ciktiSayisi];

            
            for (int i = 0; i < gizliSayisi; i++)
                for (int j = 0; j < girdiSayisi; j++)
                    agirlikGizli[i, j] = (rnd.NextDouble() - 0.5);

            for (int i = 0; i < ciktiSayisi; i++)
                for (int j = 0; j < gizliSayisi; j++)
                    agirlikCikti[i, j] = (rnd.NextDouble() - 0.5);

            for (int i = 0; i < gizliSayisi; i++)
                biasGizli[i] = (rnd.NextDouble() - 0.5);
            for (int i = 0; i < ciktiSayisi; i++)
                biasCikti[i] = (rnd.NextDouble() - 0.5);
        }

      
        double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        double SigmoidTurev(double y) => y * (1 - y);

     
        public double[] FeedForward(double[] inputs)
        {
            double[] h = new double[gizliSayisi];
            for (int i = 0; i < gizliSayisi; i++)
            {
                double sum = biasGizli[i];
                for (int j = 0; j < girdiSayisi; j++)
                    sum += agirlikGizli[i, j] * inputs[j];
                h[i] = Sigmoid(sum);
            }

            double[] c = new double[ciktiSayisi];
            for (int i = 0; i < ciktiSayisi; i++)
            {
                double sum = biasCikti[i];
                for (int j = 0; j < gizliSayisi; j++)
                    sum += agirlikCikti[i, j] * h[j];
                c[i] = Sigmoid(sum);
            }
            return c;
        }


        public void Train(double[][] egX, double[][] egY, int epoch)
        {
            for (int e = 0; e < epoch; e++)
            {
                for (int k = 0; k < egX.Length; k++)
                {
                    
                    double[] h = new double[gizliSayisi];
                    for (int i = 0; i < gizliSayisi; i++)
                    {
                        double sum = biasGizli[i];
                        for (int j = 0; j < girdiSayisi; j++)
                            sum += agirlikGizli[i, j] * egX[k][j];
                        h[i] = Sigmoid(sum);
                    }
                    double[] c = new double[ciktiSayisi];
                    for (int i = 0; i < ciktiSayisi; i++)
                    {
                        double sum = biasCikti[i];
                        for (int j = 0; j < gizliSayisi; j++)
                            sum += agirlikCikti[i, j] * h[j];
                        c[i] = Sigmoid(sum);
                    }

                    
                    double[] deltaC = new double[ciktiSayisi];
                    for (int i = 0; i < ciktiSayisi; i++)
                        deltaC[i] = (egY[k][i] - c[i]) * SigmoidTurev(c[i]);

                 
                    double[] deltaH = new double[gizliSayisi];
                    for (int i = 0; i < gizliSayisi; i++)
                    {
                        double err = 0;
                        for (int j = 0; j < ciktiSayisi; j++)
                            err += deltaC[j] * agirlikCikti[j, i];
                        deltaH[i] = err * SigmoidTurev(h[i]);
                    }

                   
                    for (int i = 0; i < ciktiSayisi; i++)
                    {
                        for (int j = 0; j < gizliSayisi; j++)
                            agirlikCikti[i, j] += ogrenmeOrani * deltaC[i] * h[j];
                        biasCikti[i] += ogrenmeOrani * deltaC[i];
                    }
                  
                    for (int i = 0; i < gizliSayisi; i++)
                    {
                        for (int j = 0; j < girdiSayisi; j++)
                            agirlikGizli[i, j] += ogrenmeOrani * deltaH[i] * egX[k][j];
                        biasGizli[i] += ogrenmeOrani * deltaH[i];
                    }
                }
            }
        }
    }
}
