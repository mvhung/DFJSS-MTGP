# DFJSS-MTGP

## Cấu trúc 
 - entity: khai báo các đối tượng, thuộc tính của các đối tượng trong xưởng.
 - gen_data: sử dụng các entity để tạo ra các bộ dữ liệu (điều chỉnh các tham số của xưởng tùy thuộc vào môi trường xưởng muốn mô phỏng).
 - GP: khai báo các phương thức của GP, sử dụng thư viện DEAP.
 - các file train: train theo phương thức CCGP và MTGP với các hàm mục tiêu tardiness và makespan.
 - plot: sử dụng các hàm trong các file .pkl để chạy thử xếp lịch.
### note: ký hiệu flow là biểu hiện cho makespan (do nhầm lẫn trong quá trình nghiên cứu).