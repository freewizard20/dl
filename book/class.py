class test:
	def __init__(self):
		return None;
	def hello(self,h):
		self.hello = h;
	def print(self):
		print(self.hello)

m = test();
m.hello(32);
m.print();
