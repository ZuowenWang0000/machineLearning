import java.util.Scanner;
import java.lang.Math;
import java.io.IOException;
import java.io.File;

public class SOM{
    public static void main(String[] args) throws Exception{
        File f = new File("./xy");
        Scanner sc = new Scanner(f);
        int numLine = sc.nextInt();

        double aINI = 0.3;
        double aneighINI = 0.2;

        //samples
        double[][] s = new double[20][2];
        for(int i = 0; i < numLine; i ++ ){
            for(int j = 0; j < 2; j++){
                s[i][j] = sc.nextDouble();
            }
        }
        sc.close();
        //finished loading the sample points into this 2 dimensional array
        File f2 = new File("./centroids");
        Scanner sc2 = new Scanner(f2);
        //centroids
        double[][] c = new double[4][2];
        for(int i = 0; i < 4; i ++ ){
            for(int j = 0; j < 2; j++){
                c[i][j] = sc2.nextDouble();
            }
        }
        sc2.close();

        int iteration = 0;
        while(iteration < 30){ //#iteration
            System.out.println("ITERATION: " + iteration);
            //for part b there should be no change in a and aneigh in different iteration rounds.
            double a = aINI - 0.02*iteration; 
            double aneigh = aneighINI - 0.02*iteration;
            a = Math.max(a, 0);
            aneigh = Math.max(aneigh, 0);

            double forCompareX = c[0][0];
            double forCompareY = c[0][1];

            //for each sample node
            for(int i = 0; i < numLine; i++){
                //for each centroid
                double sx = s[i][0];
                double sy = s[i][1];
                int minIndex = 0;
                double tempMinDis = D(sx, sy, c[0][0], c[0][1]);
                //System.out.println("tempMinDis = "+ tempMinDis);
                for(int j = 1; j < 4; j++){
                    if(D(sx, sy, c[j][0], c[j][1]) < tempMinDis){
                        tempMinDis = D(sx, sy, c[j][0], c[j][1]);
                        minIndex = j;
                        //System.out.println("minIndex = " + minIndex);
                    }
                }

                //update closest centroid and its neighbors
                c[minIndex][0] = c[minIndex][0] + a*(sx - c[minIndex][0]);
                c[minIndex][1] = c[minIndex][1] + a*(sy - c[minIndex][1]);
                
                for(int j = 0; j < 4; j++){
                    if(j == minIndex){
                        break;
                    }
                    c[j][0] = c[j][0] + aneigh*(sx - c[j][0]);
                    c[j][1] = c[j][1] + aneigh*(sy - c[j][1]);
                }
            }
            for(int k = 0; k < 4; k++){
                System.out.println("(" + c[k][0]+" , " + c[k][1] + ")");
            }
            System.out.println("diffX " + (c[0][0]-forCompareX) + "  diffY " + ((c[0][1]-forCompareY)));
            iteration ++;}
    }
    //compute the euclidean distance between two points
    public static double D(double x1, double y1, double x2, double y2){
        return Math.sqrt(Math.pow((x1-x2), 2) + Math.pow((y1-y2), 2));
    }
}