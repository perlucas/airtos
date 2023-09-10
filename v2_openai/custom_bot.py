from math import ceil


class CustomBot:
    def __init__(self, initial_stock_price, initial_cash, initial_stocks):
        self.stock_price = initial_stock_price
        self.cash = initial_cash
        self.stocks = initial_stocks

    def operate(self, operation, num_stocks):
        if operation == 'N' or num_stocks <= 0:
            return

        if operation == 'B':
            n = num_stocks
            while n * self.stock_price > self.cash and n > 0:
                n -= 1
            if n > 0:
                self.stocks += n
                self.cash -= self.stock_price * n
        else:
            n = num_stocks
            while n > self.stocks and n > 0:
                n -= 1
            if n > 0:
                self.stocks -= n
                self.cash += self.stock_price * n

    def update_stock_price(self, variations):
        for v in variations:
            self.stock_price = self.stock_price * (1 + v)

    def get_funds(self):
        return self.cash + self.stock_price * self.stocks

    def process_action(self, action):
        # print(f"action is: {action}")
        op_value = action[0]
        num_stocks = ceil(action[1] * 10)

        operation = 'N'
        if op_value > 0.71:
            operation = 'B'
        elif op_value > 0.31:
            operation = 'S'

        # print(f"op: {operation}, N: {num_stocks}")
        self.operate(operation, num_stocks)
