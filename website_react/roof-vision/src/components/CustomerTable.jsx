function CustomerTable() {
    let table = [];
    
    for (let i = 0; i < 10; i++) {
        table.push(
            <div>
                <h1>Message {i}</h1>
                <p>This is a message.</p>
            </div>
        );
    }

    return <div className="CustomerTable">{table}</div>;
}

export default CustomerTable