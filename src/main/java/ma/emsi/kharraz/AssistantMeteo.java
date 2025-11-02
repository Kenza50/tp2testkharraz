package ma.emsi.kharraz;

import dev.langchain4j.service.SystemMessage;

public interface AssistantMeteo {
    @SystemMessage("""
            Vous êtes un assistant météo.
            """)
    String chat(String userMessage);
}
