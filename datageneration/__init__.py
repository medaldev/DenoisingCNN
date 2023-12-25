# void InitialK(int N, complex *K, complex *W)
# {
# 	int	index, ind, i, j, i1, i2, i3, j1, j2, j3, ii1, ii2, ii3;
# 	int	 x,y,p1, p2, p3, p4, p5, p6, p7, p8;
# 	complex s = 0.0;
#
# 	for (i1 = 0; i1 < N; i1++){
# 		for (i2 = 0; i2 < N; i2++){
# 			//K[] = ;
# 		}
# 	}
#
# 	int r = p ;
# 	cout << r << endl;
#
# 	cout << "n_x  " << n_x << endl;
# 	for (i = 0; i <n_x; i++) {
# 		for (j = 0; j <n_x; j++) {
# 			x = i - r;
# 			y = j - r;
#
# 			//cout << x << "\t" << y << endl;
# 	//system("pause");
#
# 			if (x * x + y * y < r * r) {
# 				K[N1 + n_x*i + j] = 50.0;
# 				std::cout << "+ ";
# 			}
# 			else {
# 				K[N1 + n_x*i + j] = k1;
# 				cout << ". ";
# 			}
# 		}
#
# 		cout << endl;
# 	}
# 	system("pause");
# 	ind = 0;
# 	for (i2 = 0; i2 < num_y; i2++){
# 		for (i3 = 0; i3 < num_x; i3++){
#
# 				p1 = N1  + 2 * i2*n_x + 2 * i3;
#
# 				p2 = p1 + 1;
# 				p3 = p1 + n_x;
# 				p4 = p1 + n_x + 1;
#
# 				s = K[p1] + K[p2] + K[p3] + K[p4] ;
# 				K[ind] = s / ((double)(point*point));
#
# 				ind++;
# 		}
# 	}
#
# 	for (i1 = 0; i1 < N; i1++){
# 		//K[i1] *= W[i1];
# 	}
#
# }