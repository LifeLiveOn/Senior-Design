function CustomerTable() {
    // fetch("http://localhost:8000/customers/", {
    //     headers: {
    //         "Authorization": "Bearer " + localStorage.getItem("token")
    //     }
    // })
    // .then(res => res.json())
    // .then(data => console.log(data));

    // fetch("http://localhost:8000/login", {
    // method: "POST",
    // headers: { "Content-Type": "application/json" },
    // body: JSON.stringify({ "kassondavis@gmail.com", password })
    // })
    // .then(res => res.json())
    // .then(data => localStorage.setItem("token", data.token));

    // // ## Example: Authenticated GET

    // fetch("http://localhost:8000/customers/", {
    // headers: {
    // "Authorization": "Bearer " + localStorage.getItem("token")
    // }
    // })
    // .then(res => res.json())
    // .then(data => console.log(data));

    let rows = [];
    
    for (let i = 0; i < 10; i++) {
        rows.push(
            <tr>
                <td>2025-11-{i}</td>
                <td>hgtejyn gtfrejyt</td>
                <td>1234 fdsafd fds</td>
                <td>1234567890</td>
            </tr>
        );
    }

    return (
        <table className="CustomerTable">
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Name</th>
                    <th>Address</th>
                    <th>Number</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

    );
}

export default CustomerTable