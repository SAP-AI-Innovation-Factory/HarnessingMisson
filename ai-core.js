const axios = require('axios');
// Define the destination created in BTP cockpit
const AI_CORE_DESTINATION = "PROVIDER_AI_CORE_DESTINATION_HUB";
// Define the API Version of the LLM model
const AI_API_VERSION = "2023-05-15";
// Client credentials
const clientId = "{Client ID}";
const clientSecret = "{Cliente Secret}";
const tokenUrl = "{URL + /oauth/token}";
const resourceGroupId = "default";
 // Enter the deployment id associated to the embedding model
 const embedDeploymentIdGenAI = "<Your DeploymenT ID>";
 // Enter the deployment id associated to sentiment defined in Gen AI hub.
 const sentimentDeploymentIdGenAI = "<Your DeploymenT ID>";
// Function to get Bearer Token using clientId and clientSecret
async function getToken() {
    
    const data = new URLSearchParams({
        grant_type: 'client_credentials',
        client_id: clientId,
        client_secret: clientSecret
    });
    try {
        const response = await axios.post(tokenUrl, data);
        return response.data.access_token; // This is your Bearer Token
    } catch (error) {
        console.error('Error getting token:', error);
        throw new Error('Failed to get Bearer Token');
    }
}
async function connectToGenAI(prompt) {
    try {
        // Get the Bearer Token
        const token = await getToken();
        // Set the headers with the obtained token
        const headers = {
            "Content-Type": "application/json",
            "AI-Resource-Group": resourceGroupId,
            "Authorization": `Bearer ${token}`
        };
        // Connect to the Gen AI hub destination service
        const aiCoreService = await cds.connect.to(AI_CORE_DESTINATION);
        // Get embeddings from Gen AI hub based on the prompt
        const texts = prompt;
        // Prepare the input data to be sent to Gen AI hub model
        const payloadembed = {
            input: texts
        };
        // Call Gen AI REST API via the destination
        const responseEmbed = await aiCoreService.send({
            query: `POST inference/deployments/${embedDeploymentIdGenAI}/embeddings?api-version=${AI_API_VERSION}`,
            data: payloadembed,
            headers: headers
        });
        // The embedding is retrieved from the REST API
        const input = responseEmbed["data"][0]?.embedding;
        // Get the embedding information from the vector table.
        const query = "SELECT TOP 10 FILENAME,TO_VARCHAR(TEXT) as TEXT,COSINE_SIMILARITY(VECTOR, TO_REAL_VECTOR('[" + input + "]')) as SCORING FROM REVIEWS_TARGET ORDER BY COSINE_SIMILARITY(VECTOR, TO_REAL_VECTOR('[" + input + "]')) DESC";
        const result = await cds.run(query);
        // Pass the embedding to LLM model to receive the sentiment.
        var sentimentPrompt = null;
        var payload = {};
        var sentimentResponse = {};
        var finalResponse = { Response: [] };
        // Loop through the 10 records and retrieve the sentiment associated with them
        for (let i = 0; i < 10; i++) {
            // Preparing the prompt for the model
            sentimentPrompt = "Provide sentiment in exactly one word for the:'" + result[i].TEXT + "'";
            payload = {
                model: "gpt-4",
                messages: [
                    {
                        
                        role: "user",
                        content: sentimentPrompt
                    }
                ],
                max_tokens: 60,
                temperature: 0.0,
                frequency_penalty: 0,
                presence_penalty: 0,
                stop: null
            };
            sentimentResponse = await aiCoreService.send({
                query: `POST /inference/deployments/${sentimentDeploymentIdGenAI}/chat/completions/?api-version=${AI_API_VERSION}`,
                data: payload,
                headers: headers
            });
            finalResponse.Response.push({ "FILENAME": JSON.parse(result[i].FILENAME.toString('utf-8')).doc_name, "TEXT": result[i].TEXT, "SCORING": result[i].SCORING, "SENTIMENT": sentimentResponse["choices"][0].message.content })
        }
        return finalResponse;
    } catch (error) {
        console.error('Error connecting to GenAI:', error);
        throw error;
    }
}
module.exports = { connectToGenAI };
