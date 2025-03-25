CREATE DATABASE QUANLYBANHANG
USE QUANLYBANHANG

CREATE TABLE Vattu(
	Mavtu char(4) PRIMARY KEY,
	Tenvtu varchar(100) UNIQUE,
	Dvtinh varchar(10) DEFAULT '',
	Phantram real 
	CONSTRAINT Chk_Vattu_Phantram CHECK(Phantram BETWEEN 0 AND 100))

CREATE TABLE Nhacungcap(
	Manhacc char(3) PRIMARY KEY,
	Tennhacc varchar(100) UNIQUE,
	Diachi varchar(200),
	Dienthoai varchar(20) DEFAULT 'chua co')

CREATE TABLE Dondh(
	Sodh char(4) PRIMARY KEY,
	Manhacc char(3),
	Ngaydh datetime
	)

CREATE TABLE Chitietdondh(
	Sodh char(4),
	Mavtu char(4),
	Soluongdat int CHECK (Soluongdat>0))

CREATE TABLE Nhaphang(
	Sophieunhap char(4) PRIMARY KEY,
	Ngaynhap datetime,
	Sodh char(4))

CREATE TABLE Chitietnhaphang(
	Sophieunhap char(4) ,
	Mavtu char(4),
	Soluongnhap int CHECK(Soluongnhap >0),
	Dongianhap money CHECK(Dongianhap>0))

CREATE TABLE Xuathang(
	Sophieuxuat char(4) PRIMARY KEY,
	Ngayxuat datetime,
	Tenkh varchar(100))

CREATE TABLE Chitietxuathang(
	Sophieuxuat char(4),
	Mavtu char(4),
	Soluongxuat int CHECK(Soluongxuat>0),
	Dongiaxuat money CHECK(Dongiaxuat>0))

CREATE TABLE Tonkho(
	Namthang char(6),
	Mavtu char(4),
	Soluongdau int default 0 CHECK(Soluongdau >=0), --soluongtondauky
	Tongsln int default 0 CHECK(Tongsln >=0), -- tong so luong nhap trong ky
	Tongslx int default 0 CHECK(Tongslx >=0), -- tong so luong xuat trong ky
	Soluongcuoi AS Soluongdau + Tongsln - Tongslx)-- so luong ton cuoi ky

--tao lien ket giua cac bang 
--tao contraint sau khi tao bang
-- da noi cai phan con lai trong database diagrams
ALTER TABLE Dondh  
ADD CONSTRAINT FRK_Dondh_Nhacungcap FOREIGN KEY (Manhacc)  REFERENCES Nhacungcap(Manhacc)


INSERT INTO Vattu VALUES
('DD01','Dau DVD Hitachi 1 dia','Bo',40),
('DD02','Dau DVD Hitachi 3 dia','Bo',40),
('TL15','Tu lanh Sanyo 150 lit','Cai',25),
('TL90','Tu lanh Sanyo 90 lit','Cai',20),
('TV14','Tivi Sony 14 inches','Cai',15),
('TV21','Tivi Sony 21 inches','Cai',10),
('TV29','Tivi Sony 29 inches','Cai',10),
('VD01','Dau VCD Sony 1 dia','Bo',30),
('VD02','Dau VCD Sony 3 dia','Bo',30)

INSERT INTO Nhacungcap VALUES 
('C01','Bui Tien  Truong','Xuan La, Tay Ho, Ha Noi','0989995221'),
('C02','Nguyen  Thi Thu','Quan La, Tay Ho, Ha Noi','0979012300'),
('C03','Ngo  Thanh Tung','Kim Lien, Dong Da','0988098591'),
('C04','Bui Tien  Lap','Ha Noi','0904255934'),
('C05','Hong  That Cong','Ha Noi','chua co'),
('C07','Bui Duc  Kien','To 11, Cum 2, Xuan La','0437530097')

INSERT INTO Dondh VALUES 
('D001','C03','01/15/2002'),
('D002','C01','01/30/2002'),
('D003','C02','02/10/2002'),
('D004','C05','02/17/2002'),
('D005','C02','03/01/2002'),
('D006','C05','03/12/2002')

INSERT INTO Nhaphang VALUES 
('N001','01/17/2002','D001'),
('N002','01/20/2002','D001'),
('N003','01/31/2002','D002'),
('N004','02/15/2002','D003')

INSERT INTO Chitietdondh VALUES
('D001','DD01',10),
('D001','DD02',15),
('D002','VD02',30),
('D003','TV14',10),
('D003','TV29',20),
('D004','TL90',10),
('D005','TV14',10),
('D005','TV29',20),
('D006','TV14',10),
('D006','TV29',20),
('D005','VD01',20)

INSERT INTO Chitietnhaphang VALUES
('N001','DD01',8,2500000),
('N001','DD02',10,3500000),
('N002','DD01',2,2500000),
('N002','DD02',5,3500000),
('N003','VD02',30,2500000),
('N004','TV14',5,2500000),
('N004','TV29',12,3500000)

INSERT INTO Xuathang VALUES
('X001','01/17/2002','Duong Minh Chau'),
('X002','01/25/2002','Nguyen Kim Dung'),
('X003','01/31/2002','Nguyen Tien Dung')

INSERT INTO Chitietxuathang VALUES
('X001','DD01',2,3500000),
('X002','DD01',1,3500000),
('X002','DD02',5,4900000),
('X003','DD01',3,3500000),
('X003','DD02',2,4900000),
('X003','VD02',10,3250000)

INSERT INTO Tonkho(Namthang,Mavtu,Soluongdau,Tongsln,Tongslx) VALUES
('200201','DD01',0,10,6),
('200201','DD02',0,15,7),
('200201','VD02',0,30,10),
('200202','DD01',4,0,0),
('200202','DD02',8,0,0),
('200202','VD02',20,0,0),
('200202','TV14',5,0,0),
('200202','TV29',12,0,0)

-- Tạo view vw_Vattu gồm (MaVTu và TenVTu) dùng để liệt kê danh sách các vật tư hiện có trong bảng Vattu :
CREATE VIEW vw_Vattu
AS 
	SELECT Mavtu, Tenvtu
	FROM Vattu
GO


--Tạo view vw_Dondh_Tongsoluongdathang gồm(SoHD,TongSLDat và TongSLNhap)dùng để thống kê những đơn đặt hàng đã được nhập hàng đầy đủ :


CREATE VIEW vw_Dondh_Tongsoluongdathang
AS
	SELECT c.Sodh, SUM(Soluongdat) Tongdat, SUM(Soluongnhap) Tongnhap
		FROM Chitietdondh c
		LEFT JOIN 
		(SELECT Sodh,Mavtu,SUM(Soluongnhap) Soluongnhap
		FROM Chitietnhaphang c1
		LEFT JOIN Nhaphang n
		ON n.Sophieunhap = c1.Sophieunhap
		GROUP BY Sodh,Mavtu) h
		ON c.Mavtu = h.Mavtu AND c.Sodh = h.Sodh
		group by c.Sodh
GO	

SELECT *
FROM vw_Dondh_Tongsoluongdathang

--Tạo view vw_Dondh_Danhapdu gồm (SoHD, DaNhapDu) có hai giá trị là ‘Da Nhap Du’ nếy đơn hàng đó đã nhập đủ hoặc ‘Chua Nhap Du’ nếu đơn đặt hàng đó chưa nhập đủ 
CREATE VIEW vw_Dondh_Danhapdu 
AS
	SELECT *, 
	CASE WHEN Tongnhap >= Tongdat THEN 'Da nhap du'
	ELSE 'Chua nhap du' 
	END AS Da_nhap_du
	FROM vw_Dondh_Tongsoluongdathang
GO

SELECT *
FROM vw_Dondh_Danhapdu
 select * from Chitietdondh
--Tạo view vw_Tongnhap gồm (NamThang, MaVtu và TongSLNhap) dùng để thống kê số lượng nhập của các vật tư trong từng năm tháng tương ứng.(không sử dụng tồn kho) :
CREATE VIEW vw_Tongnhap
AS
	SELECT CONVERT(CHAR(6),NgayNhap,112) AS Namthang, Mavtu, Sum(Soluongnhap) AS Tongnhap
	FROM Nhaphang n
	INNER JOIN Chitietnhaphang c
	ON n.Sophieunhap = c.Sophieunhap
	GROUP BY CONVERT(CHAR(6),NgayNhap,112),Mavtu
GO


-- Tạo view vw_Tongxuat gồm (NamThang, MaVTu và TongSLXuat) dùng để thống kê SL xuất của vật tư trong từng năm tháng tương ứng.(không sử dụng TONKHO) :

CREATE VIEW vw_Tongxuat
AS 
	SELECT CONVERT(char(6),Ngayxuat,112) Namthang, Mavtu, SUM(Soluongxuat) as Tongxuat
	FROM Xuathang x
	JOIN Chitietxuathang c
	ON x.Sophieuxuat = c.Sophieuxuat
	GROUP BY CONVERT(char(6),Ngayxuat,112),Mavtu 
GO

SELECT * FROM vw_Tongxuat
--Tạo view vw_Dondh_Mavtu_Tongsln gồm(SoHD,NgayHD,MaVTu,TenVTu,SLDatvàTongSLDaNhap)

CREATE VIEW vw_Dondh_Mavtu_Tongsln
AS
	SELECT c.Sodh,Ngaydh,v.Mavtu,Tenvtu, Soluongdat, SUM(Soluongnhap) Soluongnhap
	FROM (
	SELECT c1.Sodh, Mavtu, Soluongdat, Ngaydh 
	FROM Chitietdondh c1 
	FULL JOIN Dondh d 
	ON d.Sodh = c1.Sodh) c
	LEFT JOIN 
	(SELECT Sodh,Mavtu, SUM(Soluongnhap) Soluongnhap
	FROM Chitietnhaphang c1
	LEFT JOIN Nhaphang n
	ON n.Sophieunhap = c1.Sophieunhap
	GROUP BY Sodh, Mavtu) h
	ON c.Mavtu = h.Mavtu AND c.Sodh = h.Sodh
	LEFT JOIN Vattu v
	ON v.Mavtu = c.Mavtu
	group by c.Sodh, Ngaydh,v.Mavtu,Tenvtu, Soluongdat
GO

select * from vw_Dondh_Mavtu_Tongsln

--Danh sách các phiếu đặt hàng chưa được nhập hàng 
select *
from vw_Dondh_Tongsoluongdathang
WHERE tong_nhap IS NULL

-- Danh sách các mặt hàng chưa được đặt hàng bao giờ 
SELECT Mavtu, Tenvtu 
FROM Vattu
WHERE Mavtu NOT IN (SELECT Mavtu FROM Chitietdondh)

SELECT  v.Mavtu, v.Tenvtu
FROM Vattu v
LEFT JOIN Chitietdondh  c
ON v.Mavtu = c.Mavtu
WHERE c.Mavtu IS NULL

-- Nhà cung cấp nào có nhiều đơn đặt hàng nhất : 

SELECT top 1 WITH TIES Manhacc, COUNT(Manhacc) 
FROM Dondh 
GROUP BY Manhacc 
ORDER BY COUNT(Manhacc) desc


-- Vật tư nào có tổng số lượng xuất bán nhiều nhất 

SELECT  TOP 1 WITH TIES Mavtu, SUM(Soluongxuat) Tong_xuat
FROM Chitietxuathang
GROUP BY Mavtu
ORDER BY 2 DESC

--Cho biết đơn đặt hàng nào có nhiều mặt hàng nhất 

SELECT TOP 1 WITH TIES  Sodh, COUNT(Mavtu) Soluongmathang
FROM Chitietdondh
GROUP BY Sodh
ORDER BY 2 DESC

-- Tạo View vw_truc_TG  báo cáo Tình hình xuất nhập vật tư 
CREATE VIEW vw_Xuat_Nhap
AS 
	SELECT Ngaynhap as Ngay, SUBSTRING(n.Sophieunhap,1,1) as Sophieu,  Mavtu, Soluongnhap, Dongianhap
	FROM Nhaphang n
	FULL JOIN Chitietnhaphang c1
	ON n.Sophieunhap = c1.Sophieunhap
	UNION
	SELECT Ngayxuat, SUBSTRING(x.Sophieuxuat,1,1), Mavtu, Soluongxuat, Dongiaxuat
	FROM Xuathang x
	FULL JOIN Chitietxuathang c2
	ON x.Sophieuxuat = c2.Sophieuxuat
GO

SELECT * FROM vw_Xuat_Nhap

-- Tìm phiếu xuất có doanh thu cao nhất
SELECT Sophieuxuat, sum(Dongiaxuat*Soluongxuat)AS Doanhthu, 
RANK() OVER(ORDER BY sum(Dongiaxuat*Soluongxuat) desc)
FROM Chitietxuathang 
group by Sophieuxuat

-- Xây   dựng   thủ   tục   tìm   SL   đặt   hàng   với   tên pro_Dondh_Tinhsldat 
--với 2 tham số vào là SoHD, MaVTu và 1 tham số ra là SL đặt của mỗi vật tư 
--trong 1 số đặt hàng

CREATE PROCEDURE pro_Dondh_Tinhsldat
	@Sodh char(4),
	@Mavtu char(4),
	@Soluong int OUTPUT

AS 
	SELECT @Soluong = Soluongdat
	FROM Chitietdondh
	WHERE Sodh = @Sodh AND Mavtu = @Mavtu
GO
DECLARE  @Ra  INT 
EXECUTE  pro_Dondh_Tinhsldat 'D001','DD02',@Ra OUTPUT
PRINT  @Ra





-- Viết hàm fn_soluongtonkho in ra ton kho trong Tonkho cua mat hang nhap vao va tháng năm tương ứng

select * from Tonkho
GO 
CREATE FUNCTION fn_soluongtonkho (@mavtu char(4),@thangnam char(6))
RETURNS INT
AS 
BEGIN 
	DECLARE @soluonghienco  INT
	SELECT @thangnam = Namthang,@soluonghienco = Soluongcuoi
	FROM Tonkho
	WHERE Mavtu = @mavtu
	RETURN @soluonghienco
END
GO
drop function fn_soluongtonkho
PRINT dbo.fn_soluongtonkho('TV14',200201)

/*
 Xây   dựng   thủ   tục   thêm   dữ   liệu   vào   bảng   VATTU   với   tên Sp_VatTu_Them 
 với 4 tham số vào chính là MaVTu, TenVTu, DVTinh, PhanTram (MaVTu phải duy nhất).
 Kiểm tra ràng buộc dữ liệu phải hợp lệ trước khi thực hiện INSERT 
 */
 GO
 CREATE PROCEDURE sp_vattu_them 
 ( @mavtu char(4),
   @tenvtu varchar(100),
   @dvtinh varchar(10),
   @ptram real)
AS
BEGIN
	IF EXISTS(SELECT*FROM Vattu WHERE Mavtu = @mavtu)
	BEGIN
	PRINT N'Mã vật tư bị trùng'
	RETURN 
	END 

	IF @ptram < 0 OR @ptram > 100 
	BEGIN 
	PRINT 'Vuot qua phan tram quy dinh'
	RETURN 
	END 
	INSERT Vattu
	VALUES (@mavtu,  @tenvtu, @dvtinh, @ptram)
	
END
GO

EXEC sp_vattu_them 'DD01','Tivi', Cai,40

