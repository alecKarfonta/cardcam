# Legal Compliance and Attribution Documentation

## Overview

This document outlines the legal compliance framework for the Trading Card Image Segmentation project, ensuring proper attribution, copyright compliance, and ethical data usage across all data sources.

## Data Sources and Legal Status

### 1. Scryfall API (Magic: The Gathering)

**Legal Status**: ✅ Compliant
- **Source**: [Scryfall API](https://scryfall.com/docs/api)
- **License**: Scryfall provides free access to Magic: The Gathering card data
- **Copyright**: Card images are © Wizards of the Coast LLC
- **Usage Rights**: Educational and non-commercial research permitted
- **Attribution Required**: Yes

**Attribution Format**:
```
Card data and images courtesy of Scryfall LLC.
Magic: The Gathering cards are © Wizards of the Coast LLC.
```

**Compliance Requirements**:
- Rate limiting: 50-100ms between requests (implemented)
- No bulk downloading for commercial purposes
- Proper attribution in all derivative works
- Respect for Wizards of the Coast intellectual property

### 2. Pokémon TCG API

**Legal Status**: ✅ Compliant
- **Source**: [Pokémon TCG API](https://pokemontcg.io/)
- **License**: Free API for non-commercial use
- **Copyright**: Pokémon cards are © Nintendo/Creatures Inc./GAME FREAK inc.
- **Usage Rights**: Educational and research purposes
- **Attribution Required**: Yes

**Attribution Format**:
```
Pokémon card data provided by pokemontcg.io
Pokémon and Pokémon character names are trademarks of Nintendo.
```

**Compliance Requirements**:
- Rate limiting: 1000 requests per hour with API key
- Non-commercial use only
- Proper trademark attribution
- No redistribution of card images for commercial purposes

### 3. YGOPRODeck API (Yu-Gi-Oh!)

**Legal Status**: ✅ Compliant
- **Source**: [YGOPRODeck API](https://ygoprodeck.com/api-guide/)
- **License**: Free API access
- **Copyright**: Yu-Gi-Oh! cards are © Konami Digital Entertainment
- **Usage Rights**: Educational and research purposes
- **Attribution Required**: Yes

**Attribution Format**:
```
Yu-Gi-Oh! card data provided by YGOPRODeck
Yu-Gi-Oh! is a trademark of Konami Digital Entertainment
```

**Compliance Requirements**:
- Rate limiting: 20 requests per second maximum
- Educational/research use only
- Local hosting of images required (no hotlinking)
- Proper copyright attribution

### 4. Community Datasets

**PSA Baseball Grades Dataset**:
- **Status**: Public dataset with proper licensing
- **Usage**: Research and educational purposes
- **Attribution**: Required as per dataset license

**Kaggle Collections**:
- **Status**: Various licenses - check individual dataset terms
- **Usage**: Follow specific dataset license requirements
- **Attribution**: As required by individual dataset licenses

## Fair Use Considerations

### Research and Educational Use
This project qualifies for fair use protection under:
- **Purpose**: Non-commercial research and education
- **Nature**: Using factual data (card metadata) and small samples for ML training
- **Amount**: Limited subset of total available cards
- **Effect**: No negative impact on market for original works

### Transformative Use
The segmentation models create transformative works by:
- Extracting individual cards from multi-card images
- Creating training data for computer vision research
- Developing tools for card collectors and researchers
- Not competing with original card sales or distribution

## Data Usage Policies

### Collection Policies
1. **Rate Limiting**: Strict adherence to API rate limits
2. **Respectful Crawling**: No aggressive scraping or server overloading
3. **Attribution**: Proper credit in all uses and publications
4. **Non-Commercial**: Research and educational use only

### Storage and Processing
1. **Local Storage**: Images stored locally for processing only
2. **No Redistribution**: Original images not redistributed
3. **Derivative Works**: Only processed/segmented outputs shared
4. **Data Retention**: Raw images deleted after processing if required

### Publication and Sharing
1. **Research Papers**: Full attribution in academic publications
2. **Code Sharing**: Attribution requirements in code repositories
3. **Model Sharing**: Clear licensing for trained models
4. **Dataset Sharing**: Only metadata and annotations, not original images

## Attribution Implementation

### Code-Level Attribution
```python
# In all source files using card data
"""
Card data sources:
- Scryfall LLC (Magic: The Gathering data)
- pokemontcg.io (Pokémon TCG data)  
- YGOPRODeck (Yu-Gi-Oh! data)

All card images and names are property of their respective copyright holders.
This project is for educational and research purposes only.
"""
```

### Database Attribution
```sql
-- Attribution table in database
CREATE TABLE data_attributions (
    id SERIAL PRIMARY KEY,
    source_api VARCHAR(100),
    copyright_holder VARCHAR(200),
    license_type VARCHAR(100),
    attribution_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Output Attribution
All generated outputs must include:
```
This dataset was created using card data from:
- Scryfall (Magic: The Gathering) - © Wizards of the Coast
- Pokémon TCG API - © Nintendo/Creatures Inc./GAME FREAK inc.
- YGOPRODeck (Yu-Gi-Oh!) - © Konami Digital Entertainment

For research and educational purposes only.
```

## Compliance Monitoring

### Automated Checks
1. **Rate Limit Monitoring**: Automatic enforcement of API limits
2. **Attribution Tracking**: Ensure all data includes proper source attribution
3. **Usage Logging**: Track all API calls and data usage
4. **License Validation**: Regular review of source license terms

### Manual Reviews
1. **Quarterly License Review**: Check for changes in API terms
2. **Attribution Audit**: Verify proper attribution in all outputs
3. **Fair Use Assessment**: Regular evaluation of fair use compliance
4. **Legal Updates**: Monitor changes in copyright law and precedent

## Risk Mitigation

### Legal Risks
- **Copyright Infringement**: Mitigated by fair use, attribution, and non-commercial use
- **Terms of Service Violation**: Prevented by strict API compliance
- **Trademark Issues**: Addressed through proper attribution and disclaimers

### Technical Safeguards
- **Rate Limiting**: Implemented in all API clients
- **Error Handling**: Graceful handling of API restrictions
- **Audit Logging**: Complete record of all data access
- **Data Lifecycle**: Clear policies for data retention and deletion

## Contact and Reporting

### Legal Questions
For legal questions or concerns:
- Review this document and source API terms
- Consult with legal counsel if needed
- Contact API providers for clarification

### Compliance Issues
To report compliance issues:
1. Document the specific concern
2. Reference relevant legal requirements
3. Propose corrective actions
4. Implement fixes and verify compliance

## Updates and Maintenance

This document will be updated:
- When new data sources are added
- When API terms of service change
- When legal requirements evolve
- At least annually for general review

**Last Updated**: 2025-09-26
**Next Review**: 2026-03-26

---

**Disclaimer**: This document provides guidance based on current understanding of applicable laws and API terms. It does not constitute legal advice. Consult with qualified legal counsel for specific legal questions.
