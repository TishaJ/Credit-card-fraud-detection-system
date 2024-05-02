// Simulate keyup event for card number input
function testCardNumberFormatting() {
    const cardNumInput = document.createElement("input");
    cardNumInput.id = "cardNum";
    document.body.appendChild(cardNumInput);

    cardNumInput.value = "1234567812345678";
    cardNumInput.dispatchEvent(new Event("keyup"));

    let expected = "1234 5678 1234 5678";
    let actual = cardNumInput.value;

    console.assert(actual === expected, `Expected: ${expected}, Actual: ${actual}`);

    console.log("Test Case 1: Validate card number formatting -", actual === expected ? "PASS" : "FAIL");

    // Cleanup
    document.body.removeChild(cardNumInput);
}

function testCardNumberValidity() {
    const cardNumInput = document.createElement("input");
    cardNumInput.id = "cardNum";
    document.body.appendChild(cardNumInput);

    cardNumInput.value = "abc123";
    cardNumInput.dispatchEvent(new Event("keyup"));

    let expected = "abc123"; // No spaces since it's non-numeric
    let actual = cardNumInput.value;

    console.assert(actual === expected, `Expected: ${expected}, Actual: ${actual}`);

    console.log("Test Case 2: Check card number validity -", actual === expected ? "PASS" : "FAIL");

    document.body.removeChild(cardNumInput);
}

window.onload = function() {
    testCardNumberFormatting();
    testCardNumberValidity();
};


    // Function to extract a parameter from the current URL by its name
    function getParameterByName(name, url = window.location.href) {
        // Escape special characters in the parameter name
        name = name.replace(/[\[\]]/g, '\\$&');

        // Created a regular expression to find the parameter in the URL
        const regex = new RegExp('[?&]' + name + '(=([^&#]*)|&|#|$)');
        const results = regex.exec(url); // Execute the regex on the URL
        
        // If no results, return null
        if (!results) return null;

        // If the parameter has no value, return an empty string
        if (!results[2]) return '';

        // Decode and return the parameter's value
        return decodeURIComponent(results[2].replace(/\+/g, ' '));
    }

    