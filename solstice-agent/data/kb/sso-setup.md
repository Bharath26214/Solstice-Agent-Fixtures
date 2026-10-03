# Single sign-on (SSO)

SAML 2.0 single sign-on is available on the Enterprise plan only. Solstice supports Okta, Microsoft Entra ID (Azure AD), Google Workspace, and any generic SAML 2.0 identity provider.

## Setting up SSO with Okta

1. In Solstice, go to Settings → Security → Single sign-on and click "Configure SAML".
2. Copy the ACS URL and the Entity ID shown there.
3. In Okta, create a new SAML 2.0 app integration and paste the ACS URL as the Single Sign-On URL and the Entity ID as the Audience URI.
4. Map the attributes: email to user.email, firstName to user.firstName, lastName to user.lastName.
5. Back in Solstice, paste Okta's metadata URL and click "Verify connection". A test login window opens; complete it with an Okta account that exists in the workspace.
6. Once verified, choose an enforcement mode: "Optional" lets members use passwords or SSO, "Required" forces SSO for everyone except designated break-glass admins.

## SCIM provisioning

SCIM 2.0 user provisioning is included with Enterprise and requires SSO to be verified first. Generate a SCIM token under Settings → Security → Provisioning; tokens expire after 12 months. Deactivating a user in your IdP deprovisions them in Solstice within 15 minutes.

## Troubleshooting

The most common failure is a clock skew above 5 minutes between the IdP and our servers, which invalidates SAML assertions. The second most common is a missing email attribute mapping. The error reference code shown on the failed login page can be given to support for a fast diagnosis.
