def check_cost_cg(self):
    holding_cost_rmp = 0.0
    transportation_cost_rmp = 0.0
    self.rmp_oracle.obj["holding_cost"] = {}
    for t in self.rmp_oracle.obj["holding_cost"].keys():
        holding_cost_rmp += self.rmp_oracle.obj["holding_cost"][t].getExpr().getValue()
    for t in self.rmp_oracle.obj["transportation_cost"].keys():
        if type(self.rmp_oracle.obj["transportation_cost"][t]) is not float:
            transportation_cost_rmp += (
                self.rmp_oracle.obj["transportation_cost"][t].getExpr().getValue()
            )
    print("tr_rmp", transportation_cost_rmp)
    transportation_cost_from_customer = 0.0
    unfulfilled_cost_from_customer = {t: 0 for t in range(self.arg.T)}
    variables = self.rmp_model.getVars()
    for v in variables:
        if v.getName().startswith("lambda"):
            # print(self.iter,v.getName())
            # print(v.getName().split('_')[1])
            for customer in self.columns.keys():
                if v.getName().split("_")[1] == str(customer):
                    print(v.getName())
                    print("lambda optimal value", v.x)
                    col_num = int(v.getName().split("_")[2])
                    print(self.columns[customer][col_num]["unfulfilled_demand_cost"])
                    print(self.columns[customer][col_num]["transportation_cost"])
                    # print("transportation_cost", self.columns[customer][0]['transportation_cost'])
                    # print("unfulfilled_demand_cost", self.columns[customer][0]['unfulfilled_demand_cost'])
                    transportation_cost_from_customer += (
                        v.x * self.columns[customer][col_num]["transportation_cost"]
                    )
                    for t in range(self.arg.T):
                        unfulfilled_cost_from_customer[t] += (
                            v.x
                            * self.columns[customer][col_num][
                                "unfulfilled_demand_cost"
                            ][t]
                        )
    print("tr_pricing", transportation_cost_from_customer)
    print(
        "transportation_cost",
        transportation_cost_rmp + transportation_cost_from_customer,
    )
    print("holding_cost", holding_cost_rmp)
    for t in range(self.arg.T):
        print(
            "unfulfilled_demand_cost",
            t,
            unfulfilled_cost_from_customer[t],
        )
    print("unfulfilled_demand_cost", unfulfilled_cost_from_customer)
