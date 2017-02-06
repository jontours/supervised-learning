d1=read.table("student/student-mat.csv",sep=";",header=TRUE)
d2=read.table("student/student-por.csv",sep=";",header=TRUE)
d3=merge(d1,d2)
#d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
print(nrow(d3)) # 382 students
print(nrow(d1))
print(nrow(d2))
