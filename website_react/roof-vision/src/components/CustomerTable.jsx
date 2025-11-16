import { useEffect, useState } from "react";

function CustomerTable() {
    const [customers, setCustomers] = useState([]);
    
    useEffect(() => {   
        fetch("http://localhost:8000/customers/", {
            credentials: "include"
        })
        .then(res => res.json())
        .then(data => {
            console.log(data);
            setCustomers(data)
        });
    }, []);

    return (
        <table className="CustomerTable">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Name</th>
                    <th>Number</th>
                </tr>
            </thead>
            <tbody> {
                customers.map((customer) => (
                    <tr key={customer.id}>
                        <td>{customer.created_at.substring(0, 10)}</td>
                        <td>{customer.name}</td>
                        <td>{customer.phone}</td>
                    </tr>
                ))
            }
            </tbody>
        </table>

    );
}

export default CustomerTable