#include <stdio.h>     // Import fundamental functions like printf()
   #include <stdlib.h>     // Import srand() function
   #include <time.h>     // Import time() function
   #include <math.h>

   // The global variables has 0 value in default
   int NODE[4][1001];
   int tmp, i, j;
   int cnt_success, cnt_arrive;     //�����������޴º�����
				    //Count on Frames arriving at slots

   double avr_success[500];     //����������հ����޴º�������
   double input_G[50];     //G��
   double A;     //G�������ذ���A �����ޱ����Ѻ�������

   int q, s;
   int ceilA;

   int main()
   {
	    FILE* fout = fopen("result_policy1.csv", "w");
	   fprintf(fout, "Average of Success Frame No,Suc_Average\n");

	   srand(time(NULL));     //rand ������
	   for(i=0; i<50; i++){
		   input_G[i] = 0.1F*(i+1);
	   }

		 for(s=0; s<50; s++){     //Simulation ����

		   tmp=0;
		   A=4.0F*(2.0F/input_G[s]);     //G�����̿��Ͽ�A��������Ѵ�
		   ceilA = ceil(A);
		   for(q=0;q<50;q++){     //50����������հ�����ϴ±���
			   for(i=0;i<4;i++){     //station��5�������ϴ±���
				   while(1){
					   tmp=tmp+rand() % ceilA +1;
					   //NODE[i][0]����[i][1000]��������
					   if(tmp>1000)
					   //NODE[4][999] �̻��̸����������
						   break;
					   NODE[i][tmp]=1;
					   //NODE[0][0]����NODE[9][999]����
					   //Frame�����۵Ǵ°���1��ǥ��
				   }
				   tmp = 0;     //tmp���ʱ�ȭ���ش�
			   }
			   for(i=0;i<1000;i++){
				   cnt_arrive=0;
				   //i �ð������۹�����������
				   for(j=0;j<4;j++){     //��station üũ�ϱ�
					   if(NODE[j][i]>0)
						   cnt_arrive++;
				   }
				//��slot��Frame�̵ΰ��̻�
				//�õ��Ǵ°���üũ�ϱ�����
				   if(cnt_arrive == 1){
                                //��slot��Frame��1�������۵ȴٸ�
					   cnt_success++;     //Fraem ���ۼ���
				   }
			   }
			   avr_success[s]+=cnt_success;     //��������50�����Ѵ�
			   cnt_success = 0;		    //���������ʱ�ȭ�Ѵ�
			   for(i=0; i<4; i++){		    //NODE���ʱ�ȭ���ش�
				   for(j=0; j<1000; j++){
					   NODE[i][j] = 0;
				   }
			   }
		   }     //��հ�����������Ѵ�
		   avr_success[s]/=50.0F;
                         //50����������������50���γ�������հ������

                         //����������50���ݺ��ϹǷ�50���γ�������հ������

		   printf("[%.1f] Average of Success Frame No. = %lf\n", ((float)s/10.0+0.1), avr_success[s]);
		   fprintf(fout, "%.1f,%lf\n", ((float)s/10.0+0.1), avr_success[s]);
	 }     //Simulation termination
	   fclose(fout);
	   return 0;
   }
